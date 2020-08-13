import torch.optim as optim
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from allennlp.modules.elmo import batch_to_ids, Elmo
import torch
import random
import string
from nltk.corpus import stopwords
import copy
import torch.nn as nn

stop_words = set(stopwords.words("english"))
for punctuation in string.punctuation:
    stop_words.add(punctuation)


def clones(nn_module, num_copies):
    return nn.ModuleList([copy.deepcopy(nn_module) for _ in range(num_copies)])


def get_optimizer(optimizer, model, lr, momentum=0, weight_decay=0):
    if type(model) == list:
        parameters = []
        for m in model:
            parameters = parameters + list(m.parameters())
    else:
        parameters = model.parameters()
    if optimizer == "SGD":
        optimizer = optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = optim.Adam(params=parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params=parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RMSprop":
        optimizer = optim.RMSprop(params=parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adamax":
        optimizer = optim.Adamax(params=parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer: {}".format(optimizer))
    return optimizer


def binarize_dataset(threshold, training_users, training_items, training_ratings):
    for i in range(len(training_ratings)):
        if training_ratings[i] > threshold:
            training_ratings[i] = 1
        else:
            training_ratings[i] = 0
    training_users = [training_users[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_items = [training_items[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_ratings = [rating for rating in training_ratings if rating != 0]
    return training_users, training_items, training_ratings


def convert_to_rating_matrix(num_users, num_items, rating_indices, rating_scores, user):
    # for rating_indices and rating_scores
    rating_matrix = np.zeros((num_users, num_items))
    if user:
        for i in range(len(rating_indices)):
            for j in range(len(rating_indices[i])):
                rating_matrix[i, rating_indices[i][j]] = rating_scores[i][j]
        return rating_matrix
    else:
        for i in range(len(rating_indices)):
            for j in range(len(rating_indices[i])):
                rating_matrix[rating_indices[i][j], i] = rating_scores[i][j]
        return rating_matrix.T


def convert_to_rating_matrix_from_lists(num_users, num_items, user_list, item_list, rating_list, user, sparse=False):
    rating_matrix = np.zeros((num_users, num_items))
    assert len(user_list) == len(item_list)
    assert len(user_list) == len(rating_list)
    for i in range(len(user_list)):
        rating_matrix[user_list[i], item_list[i]] = rating_list[i]
    if sparse:
        rating_matrix = csr_matrix(rating_matrix)
    if user:
        return rating_matrix
    else:
        return rating_matrix.T


def get_pmi_matrix_gpu(matrix, root):
    import cupy as cp
    rows, cols = matrix.shape
    item_rated = cp.array(matrix.sum(axis=0))
    pmi_matrix = []
    for i in tqdm(range(rows)):
        row_index, col_index = matrix[i].nonzero()
        if len(row_index) > 0:
            values = cp.asarray(item_rated[:, col_index]).flatten()
            values = cp.maximum(cp.log(rows/cp.power(values, root)), 0)
            pmi_matrix.append(sparse.coo_matrix((cp.asnumpy(values), (row_index, col_index)), shape=(1, cols)))
        else:
            pmi_matrix.append(sparse.coo_matrix((1, cols)))
    return sparse.vstack(pmi_matrix)


def generate_nce_matrix(matrix, root, binarize_threshold, sparse=False):
    if binarize_threshold != -1:
        matrix = np.where(matrix > binarize_threshold, 1, 0)
    matrix = csr_matrix(matrix)
    nce_matrix = get_pmi_matrix_gpu(matrix, root)
    if sparse:
        return csr_matrix(nce_matrix)
    else:
        return nce_matrix.toarray()


def evaluate_mse(prediction, user_list, item_list, rating_list):
    mse = 0
    for i in range(len(rating_list)):
        mse += (prediction[user_list[i], item_list[i]] - rating_list[i]) ** 2
    mse /= len(rating_list)
    return mse


def calculate_mse(prediction, label, mode):
    if mode == 1:
        mask = (label > 0)
        loss = (((prediction - label) * mask.float()) ** 2).sum() / mask.sum()
    elif mode == 2:
        mask = label
        count = (label > 0).sum()
        loss = (((prediction - label) * mask.float()) ** 2).sum() / count
    elif mode == 3:
        mask = label
        count = (label > 0).sum()
        loss = (((prediction - label) ** 2) * mask.float()).sum() / count
    elif mode == 0:
        loss = torch.mean((prediction - label) ** 2)
    else:
        raise NotImplementedError
    return loss


def get_batch_new(dimension, batch_size, sample_strategy, count):
    if sample_strategy == 'fixed_batch':
        return np.arange(count*batch_size, min((count + 1)*batch_size, dimension))
    elif sample_strategy == 'random_batch':
        total_num_batches = int(dimension/batch_size) + 1
        batch_index = np.random.choice(total_num_batches)
        return np.arange(batch_index*batch_size, min((batch_index + 1)*batch_size, dimension))
    elif sample_strategy == 'random':
        indices = np.random.choice(dimension, batch_size)
        return indices


def get_batches(rating_matrix, batch_size):
    t1 = time.time()
    remaining_size = rating_matrix.shape[0]
    batch_index = 0
    batches = []
    if batch_size == -1:
        batches.append(rating_matrix)
        return batches
    while remaining_size > 0:
        if remaining_size < batch_size:
            batches.append(rating_matrix[batch_index * batch_size:])
        else:
            batches.append(rating_matrix[batch_index * batch_size:(batch_index + 1) * batch_size])
        batch_index += 1
        remaining_size -= batch_size
    t2 = time.time()
    print('finished generating batches in {0} seconds'.format(t2 - t1))
    return batches


def get_batches_ids(rating_matrix, batch_size, indices):
    t1 = time.time()
    batches = rating_matrix[indices]
    t2 = time.time()
    print('finished generating batches in {0} seconds'.format(t2 - t1))
    return batches


def generate_features(max_len, lang_feature_batch_size, elmo_embedding_size, model, elmo, model_device, elmo_device, review_documents, word_dict, store_type='numpy'):
    t1 = time.time()
    model.eval()
    if elmo_embedding_size:
        elmo.eval()
    predicted_features = []
    for batch in generate_batches(review_documents, None, model_device, word_dict, max_len, lang_feature_batch_size, False):
        raw_texts, word_indices, word_mask, manual_mask, index_pointers, target_features, user_indices = batch
        current_user = 0
        if elmo_embedding_size:
            elmo_inputs = batch_to_ids(raw_texts).cuda(elmo_device)
            elmo_representations = elmo(elmo_inputs)["elmo_representations"][0]
            elmo_representations = elmo_representations.cuda(model_device)
        else:
            elmo_representations = None
        context_encoded = model.forward(word_indices=word_indices, word_mask=word_mask, manual_mask=manual_mask, elmo_representations=elmo_representations)
        for i in range(len(index_pointers)):
            if i == len(index_pointers) - 1:
                temp_features = context_encoded[index_pointers[i]:]
            else:
                temp_features = context_encoded[index_pointers[i]:index_pointers[i+1]]
            if store_type == 'numpy':
                predicted_features.append(torch.mean(temp_features, dim=0).cpu().detach().numpy())
            elif store_type == 'torch_cpu':
                predicted_features.append(torch.mean(temp_features, dim=0).cpu().detach().numpy())
                predicted_features[-1] = torch.from_numpy(predicted_features[-1]).float().unsqueeze(0)
            else:
                predicted_features.append(torch.mean(temp_features, dim=0))
            current_user += 1
    t2 = time.time()
    print('finished generating features in {0} seconds'.format(t2 - t1))
    if store_type != 'numpy':
        return torch.cat(predicted_features, dim=0)
    return predicted_features


def generate_features_transformer(max_len, lang_feature_batch_size, elmo_embedding_size, transformer, elmo, model_device, elmo_device, review_documents, word_dict, store_type='numpy'):
    transformer.eval()
    if elmo_embedding_size:
        elmo.eval()
    predicted_features = []
    for batch in generate_batches(review_documents, None, model_device, word_dict, max_len, lang_feature_batch_size, False):
        raw_texts, word_indices, word_mask, manual_mask, index_pointers, target_features, user_indices = batch
        current_user = 0
        if elmo_embedding_size:
            elmo_inputs = batch_to_ids(raw_texts).cuda(elmo_device)
            elmo_representations = elmo(elmo_inputs)["elmo_representations"][0]
            elmo_representations = elmo_representations.cuda(model_device)
        else:
            elmo_representations = None
        word_indices_transposed = word_indices.permute(1, 0)
        _, context_encoded = transformer(word_indices_transposed, True, word_mask)

        # masks = (1 - word_mask).unsqueeze(2).repeat(1, 1, args.num_topics)
        # feature_sum = torch.sum(context_encoded*masks.float(), dim=1)
        # lengths = (word_mask.shape[1] - torch.sum(word_mask, dim=1))
        # context_encoded = feature_sum / lengths.unsqueeze(1).repeat(1, args.num_topics).float()

        for i in range(len(index_pointers)):
            if i == len(index_pointers) - 1:
                temp_features = context_encoded[index_pointers[i]:]
            else:
                temp_features = context_encoded[index_pointers[i]:index_pointers[i+1]]
            if store_type == 'numpy':
                predicted_features.append(torch.mean(temp_features, dim=0).cpu().detach().numpy())
            elif store_type == 'torch_cpu':
                predicted_features.append(torch.mean(temp_features, dim=0).cpu().detach().numpy())
                predicted_features[-1] = torch.from_numpy(predicted_features[-1]).float().unsqueeze(0)
            else:
                predicted_features.append(torch.mean(temp_features, dim=0))
            current_user += 1
    if store_type != 'numpy':
        return torch.cat(predicted_features, dim=0)
    return predicted_features


def generate_batches(review_documents, latent_factors, model_device, word_dict, max_len, batch_size, shuffle):
    indices = list(range(len(review_documents)))
    if shuffle:
        random.shuffle(indices)
    for start_index in range(0, len(indices), batch_size):
        end_index = min(start_index + batch_size, len(indices))
        raw_texts = []
        index_pointers = []
        current_pointer = 0
        for index in indices[start_index:end_index]:
            raw_texts.extend([temp_review[:max_len] for temp_review in review_documents[index]])
            index_pointers.append(current_pointer)
            current_pointer += len(review_documents[index])
        # max_length = max([len(temp_review) for temp_review in raw_texts])
        max_length = max_len
        word_indices = np.zeros(shape=(len(raw_texts), max_length), dtype=int)
        word_mask = torch.ByteTensor(len(raw_texts), max_length).fill_(1).cuda(model_device)
        manual_mask = torch.ByteTensor(len(raw_texts), max_length).fill_(1).cuda(model_device)
        for i, temp_review in enumerate(raw_texts):
            offset = len(temp_review)
            word_indices[i, :offset] = [word_dict.get(word, 0) for word in temp_review]  # each row is the indices of the word in the word embedding dict
            word_mask[i, :offset].fill_(0)
            manual_mask[i, :offset].fill_(0)
            for k in range(offset):
                if temp_review[k] in stop_words:
                    manual_mask[i, k].fill_(0)
        word_indices = torch.LongTensor(word_indices).cuda(model_device)
        if latent_factors is not None:
            if type(latent_factors) != torch.Tensor:
                target_features = torch.FloatTensor([latent_factors[index] for index in indices[start_index:end_index]]).cuda(model_device)
            else:
                target_features = torch.cat([latent_factors[index, :].unsqueeze(0) for index in indices[start_index:end_index]], dim=0)
        else:
            target_features = None

        user_indices = np.array(indices)[list(range(start_index, end_index))]
        yield raw_texts, word_indices, word_mask, manual_mask, index_pointers, target_features, user_indices


def generate_batches_ids(review_documents, word_dict, max_len, indices):
    raw_texts = []
    index_pointers = []
    length_lists = []
    current_pointer = 0
    for index in indices:
        raw_texts.extend([temp_review[:max_len] for temp_review in review_documents[index]])
        index_pointers.append(current_pointer)
        current_pointer += len(review_documents[index])
        length_lists.append(len(review_documents[index]))
    # max_length = max([len(temp_review) for temp_review in raw_texts])
    max_length = max_len
    word_indices = np.zeros(shape=(len(raw_texts), max_length), dtype=int)
    word_mask = torch.ByteTensor(len(raw_texts), max_length).fill_(1).cuda()
    for i, temp_review in enumerate(raw_texts):
        offset = len(temp_review)
        word_indices[i, :offset] = [word_dict.get(word, 0) for word in temp_review]  # each row is the indices of the word in the word embedding dict
        word_mask[i, :offset].fill_(0)
    word_indices = torch.LongTensor(word_indices).cuda()

    user_indices = np.array(indices)
    return raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists

def generate_metadata(training_users, training_items, training_ratings, num_users, num_items):
    user_indices, item_indices = [[] for _ in range(num_users)], [[] for _ in range(num_items)]
    user_scores, item_scores = [[] for _ in range(num_users)], [[] for _ in range(num_items)]
    user_dict, item_dict = [dict() for _ in range(num_users)], [dict() for _ in range(num_items)]
    for k in range(len(training_ratings)):
        user = training_users[k]
        item = training_items[k]
        rating = training_ratings[k]
        user_dict[user][item] = rating
        item_dict[item][user] = rating
    for i in range(len(user_dict)):
        temp_dict = user_dict[i]
        for key in sorted(temp_dict):
            user_indices[i].append(key)
            user_scores[i].append(temp_dict[key])
    for j in range(len(item_dict)):
        temp_dict = item_dict[j]
        for key in sorted(temp_dict):
            item_indices[j].append(key)
            item_scores[j].append(temp_dict[key])
    return user_indices, item_indices, user_scores, item_scores  # user_indices = [a, b], where a is a list of items rated by user 0


def calculate_average_features(features, index_pointers):
    predicted_features = []
    for i in range(len(index_pointers)):
        if i == len(index_pointers) - 1:
            temp_features = features[index_pointers[i]:]
        else:
            temp_features = features[index_pointers[i]:index_pointers[i + 1]]
        predicted_features.append(torch.mean(temp_features, dim=0, keepdim=True))
    predicted_features = torch.cat(predicted_features, dim=0)
    return predicted_features



