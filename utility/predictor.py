import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
from multiprocessing import Process
import time
import copy
import math
from utility.model_helper import get_batch_new


def predict_old(matrix_U, matrix_V, topK, matrix_Train, bias=None, measure="Cosine", gpu=True):
    gpu = False
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)
    prediction = []

    for user_index in tqdm(range(matrix_U.shape[0])):
        vector_u = matrix_U[user_index]
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def predict_subroutine(matrix_U, matrix_V, bias, topK, matrix_Train, measure="Cosine", gpu=True):
    if matrix_V is not None or bias is not None:
        return predict_old(matrix_U, matrix_V, topK, matrix_Train, bias=bias, measure=measure, gpu=gpu)
    else:
        # print('treating matrix_U as prediction')
        # t1 = time.time()
        non_zero_indices = matrix_Train.nonzero()
        matrix_U[non_zero_indices[0], non_zero_indices[1]] = 0
        ind = np.argpartition(matrix_U, -topK, axis=1)[:, -topK:]
        arange_indices = np.expand_dims(np.arange(matrix_U.shape[0]), axis=1)
        ind = ind[arange_indices, np.argsort(matrix_U[arange_indices, ind], axis=1)]
        ind = ind[:, ::-1]
        # print('done processing prediction in {0} seconds'.format(time.time() - t1))
        return ind


def predict(matrix_U, matrix_V, bias, topK, matrix_Train, measure="Cosine", gpu=True, batch_size=5000):
    t1 = time.time()
    res = []
    print('predict in batch size {0}'.format(batch_size))
    batch_length = math.ceil(matrix_U.shape[0] / batch_size)
    for i in range(batch_length):
        indices = get_batch_new(matrix_U.shape[0], batch_size, 'fixed_batch', i)
        res.append(predict_subroutine(matrix_U[indices], matrix_V, bias, topK, matrix_Train[indices], measure, gpu))
    print('done processing prediction in {0} seconds'.format(time.time() - t1))
    return np.vstack(res)


def sub_routine(vector_u, matrix_V, vector_train, bias, measure, topK=500, gpu=True):

    train_index = vector_train.nonzero()[1]
    if measure == "Cosine":
        vector_predict = matrix_V.dot(vector_u)
    else:
        if gpu:
            import cupy as cp
            vector_predict = -cp.sum(cp.square(matrix_V - vector_u), axis=1)
        else:
            vector_predict = -np.sum(np.square(matrix_V - vector_u), axis=1)
    if bias is not None:
        if gpu:
            import cupy as cp
            vector_predict = vector_predict + cp.array(bias)
        else:
            vector_predict = vector_predict + bias

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]


def predict_batch(matrix_U, matrix_V, topK, matrix_Train, batch_size=100, bias=None, measure="Cosine", gpu=True):
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    prediction = []

    user_batch_indecs = get_batches(matrix_U, batch_size=batch_size)

    for user_batch_index in tqdm((user_batch_indecs)):
        subset_U = matrix_U[user_batch_index[0]:user_batch_index[1]]
        subset_Train = matrix_Train[user_batch_index[0]:user_batch_index[1]]

        batch_predict = batch_sub_routine(subset_U, matrix_V, subset_Train, bias, measure, topK=topK, gpu=gpu)
        prediction.append(batch_predict)

    return np.vstack(prediction)


def get_batches(rating_matrix, batch_size):
    remaining_size = rating_matrix.shape[0]
    batch_index=0
    batch_indecs = []
    while(remaining_size>0):
        if remaining_size<batch_size:
            batch_indecs.append([batch_index*batch_size,-1])
        else:
            batch_indecs.append([batch_index*batch_size,(batch_index+1)*batch_size])
        batch_index += 1
        remaining_size -= batch_size
    return batch_indecs


def batch_sub_routine(subset_U, matrix_V, subset_Train, bias, measure, topK=50, gpu=True):
    train_indecs = [i.nonzero()[1] for i in subset_Train]
    train_num_ratings = [i.nnz for i in subset_Train]
    if measure == "Cosine":
        batch_predict = subset_U.dot(matrix_V.T)
    else:
        if gpu:
            import cupy as cp
            batch_predict = (subset_U ** 2).sum(axis=-1)[:, np.newaxis] + (matrix_V ** 2).sum(axis=-1)
            batch_predict -= 2 * cp.squeeze(subset_U.dot(matrix_V[..., np.newaxis]), axis=-1)
            batch_predict **= 0.5
            batch_predict = - batch_predict
        else:
            batch_predict = (subset_U ** 2).sum(axis=-1)[:, np.newaxis] + (matrix_V ** 2).sum(axis=-1)
            batch_predict -= 2 * np.squeeze(subset_U.dot(matrix_V[..., np.newaxis]), axis=-1)
            batch_predict **= 0.5
            batch_predict = - batch_predict
    if bias is not None:
        if gpu:
            import cupy as cp
            batch_predict = batch_predict + cp.array(bias)
        else:
            batch_predict = batch_predict + bias

    if gpu:
        import cupy as cp
        candidate_indecs = cp.argpartition(-batch_predict,
                                           range(topK + max(train_num_ratings)))[:topK + max(train_num_ratings)]
        candidate_indecs = cp.asnumpy(candidate_indecs)
    else:
        candidate_indecs = np.argpartition(-batch_predict,
                                           range(topK + max(train_num_ratings)))[:topK + max(train_num_ratings)]

    batch_predict = []
    for i, vector_predict in enumerate(candidate_indecs):

        if train_num_ratings[i] > 0:
            batch_predict.append(np.delete(vector_predict,
                                           np.isin(vector_predict, train_indecs[i]).nonzero()[0])[:topK])
        else:
            batch_predict.append(np.zeros(topK))

    return np.array(batch_predict)


def sampling_predictor(matrix_U, matrix_V, topK, matrix_Train, index_Valid, bias=None, measure="Cosine", gpu=True):
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    prediction = []

    for user_index in tqdm(range(matrix_U.shape[0])):
        vector_u = matrix_U[user_index]
        vector_train = matrix_Train[user_index]
        subset_index = index_Valid[user_index]
        subset_matrix_V = matrix_V[subset_index]

        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_u, subset_matrix_V, vector_train, bias, measure, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def sampler(matrix_Valid, multiple=100):
    m, n = matrix_Valid.shape
    index_Valid = []
    for i in range(m):
        observed_index = matrix_Valid[m].nonzero()[1]
        num_observed = len(observed_index)
        if num_observed * multiple < n:
            subset_index = np.random.choice(n, num_observed * multiple, replace=False)
        else:
            subset_index = range(n)
        index_Valid.append(num_observed)
    return index_Valid
