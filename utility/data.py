from nltk.corpus import stopwords
import numpy as np
import os
import pickle
import random
import string
import torch
import time

stop_words = set(stopwords.words("english"))
for punctuation in string.punctuation:
    stop_words.add(punctuation)


def save(obj, filename, directory=None):
    if directory:
        with open(os.path.join(directory, filename), "wb+") as file_out:
            pickle.dump(obj, file_out)
    else:
        with open(filename, "wb+") as file_out:
            pickle.dump(obj, file_out)


def load(filename, directory=None):
    if directory:
        return pickle.load(open(os.path.join(directory, filename), "rb"))
    else:
        return pickle.load(open(filename, "rb"))


def load_dataset_yelp(args):
    word_dict = load("word_dict.dat", args.data_directory)
    word_embeddings = load("word_embeddings.dat", args.data_directory)
    user_documents = load("user_documents.dat", args.data_directory)
    item_documents = load("item_documents.dat", args.data_directory)
    training_users = load("training_users.dat", args.data_directory)
    training_items = load("training_items.dat", args.data_directory)
    training_ratings = load("training_ratings.dat", args.data_directory)
    validation_users = load("validation_users.dat", args.data_directory)
    validation_items = load("validation_items.dat", args.data_directory)
    validation_ratings = load("validation_ratings.dat", args.data_directory)
    return training_users, training_items, training_ratings, validation_users, validation_items, validation_ratings, user_documents, item_documents, word_dict, word_embeddings


def load_dataset_amazon(args):
    train_directory = args.data_directory + 'train/'
    test_directory = args.data_directory + 'test/'

    word_embeddings = np.load(train_directory + 'w2v.npy')
    word_embeddings = torch.from_numpy(word_embeddings)
    word_dict = load('word_dict.dat', train_directory)

    if os.path.isfile(train_directory + 'train.pkl'):
        train = load(train_directory + 'train.pkl')
        train_users, train_items, train_ratings = train
        val = load(test_directory + 'val.pkl')
        val_users, val_items, val_ratings = val
    else:
        train_indices = np.load(train_directory + 'Train.npy')
        train_ratings = np.load(train_directory + 'Train_Score.npy')
        val_indices = np.load(test_directory + 'Val.npy')
        val_ratings = np.load(test_directory + 'Val_Score.npy')
        train_users, train_items = convert_indices_to_list(train_indices)
        train_ratings = list(train_ratings)
        train = (train_users, train_items, train_ratings)
        save(train, 'train.pkl', train_directory)
        val_users, val_items = convert_indices_to_list(val_indices)
        val_ratings = list(val_ratings)
        val = (val_users, val_items, val_ratings)
        save(val, 'val.pkl', test_directory)
    if os.path.isfile(train_directory + 'user_documents.pkl'):
        user_documents = load(train_directory + 'user_documents.pkl')
        item_documents = load(train_directory + 'item_documents.pkl')
    else:
        user_review_indices = np.load(train_directory + 'userReview2Index.npy')
        user_item_review_ids = np.load(train_directory + 'user_item2id.npy')
        item_review_indices = np.load(train_directory + 'itemReview2Index.npy')
        item_user_review_ids = np.load(train_directory + 'item_user2id.npy')
        user_documents = convert_review(user_review_indices, word_dict, user_item_review_ids, len(item_user_review_ids) + 1)
        item_documents = convert_review(item_review_indices, word_dict, item_user_review_ids, len(user_item_review_ids) + 1)
        save(user_documents, 'user_documents.pkl', train_directory)
        save(item_documents, 'item_documents.pkl', train_directory)
    for i in range(len(user_documents)):
        if len(user_documents[i]) == 0:
            user_documents[i].append(['<unk>'])
    for i in range(len(item_documents)):
        if len(item_documents[i]) == 0:
            item_documents[i].append(['<unk>'])
    return train_users, train_items, train_ratings, val_users, val_items, val_ratings, user_documents, item_documents, word_dict, word_embeddings


def convert_indices_to_list(indices):
    users = []
    items = []
    for i in range(indices.shape[0]):
        users.append(indices[i, 0])
        items.append(indices[i, 1])
    return users, items


def convert_review(review_indices, word_dict, review_ids, placeholder):
    documents = []
    indices = []
    reverse_word_dict = {}
    for k in word_dict.keys():
        reverse_word_dict[word_dict[k]] = k
    for i in range(review_indices.shape[0]):
        documents.append([])
        indices.append([])
        for j in range(review_indices.shape[1]):
            if review_ids[i, j] == placeholder:
                break
            else:
                sentence = []
                indices[i].append(review_ids[i, j])
                for k in range(review_indices.shape[2]):
                    if review_indices[i, j, k] != -1:
                        sentence.append(reverse_word_dict[review_indices[i, j, k]])
                if len(sentence) > 0:
                    documents[i].append(sentence)
    return documents


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
