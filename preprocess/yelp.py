import collections
import os
import random
import sys

sys.path.append("..")

from preprocess.common import get_config, generate_embeddings, generate_metadata
from utility.data import save


def preprocess_text(text):
    text = text.strip()
    text = text.replace('<sssss>', '<EOS> <SOS>')
    text = '<SOS> ' + text + ' <EOS>'
    if text.count('SOS') != text.count('EOS'):
        print(text)
        assert 0 == 1
    return text.split()


def generate_dictionaries(args):
    user_dict, item_dict, word_dict = dict(), dict(), dict()
    word_list = []
    word_counter = [["PAD", -1], ["UNK", -1]]
    with open(os.path.join(args.data_directory, args.training_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            if line[0] not in user_dict:
                user_dict[line[0]] = len(user_dict)
            if line[1] not in item_dict:
                item_dict[line[1]] = len(item_dict)
            word_list.extend(preprocess_text(line[3]))
    if args.vocabulary_size:
        word_counter.extend(collections.Counter(word_list).most_common(args.vocabulary_size - len(word_counter)))
    else:
        word_counter.extend(collections.Counter(word_list).most_common(len(word_list)))
    if args.word_threshold:
        for word, count in word_counter:
            if count >= args.word_threshold:
                word_dict[word] = len(word_dict)
    else:
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)
    return user_dict, item_dict, word_dict


def generate_documents(args, user_dict, item_dict):
    user_documents, item_documents = [[] for _ in range(len(user_dict))], [[] for _ in range(len(item_dict))]
    user_item_review_index = [[] for _ in range(len(user_dict))]
    with open(os.path.join(args.data_directory, args.training_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            user = user_dict[line[0]]
            item = item_dict[line[1]]
            user_item_review_index[user].append(item)
            review = preprocess_text(line[3])
            user_documents[user].append(review)
            item_documents[item].append(review)
    with open(os.path.join(args.data_directory, args.validation_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            user = user_dict[line[0]]
            item = item_dict[line[1]]
            user_item_review_index[user].append(item)
            review = preprocess_text(line[3])
            user_documents[user].append(review)
            item_documents[item].append(review)
    if args.document_threshold:
        for i in range(len(user_documents)):
            temp_document = user_documents[i]
            temp_indices = user_item_review_index[i]
            mapIndexPosition = list(zip(temp_document, temp_indices))
            random.shuffle(mapIndexPosition)
            temp_document, temp_indices = zip(*mapIndexPosition)
            user_documents[i] = temp_document
            user_item_review_index[i] = temp_indices
        for temp_document in item_documents:
            random.shuffle(temp_document)
        user_documents = [temp_document[:args.document_threshold] for temp_document in user_documents]
        item_documents = [temp_document[:args.document_threshold] for temp_document in item_documents]
        user_item_review_index = [temp_indices[:args.document_threshold] for temp_indices in user_item_review_index]
    return user_documents, item_documents, user_item_review_index


def preprocess(args):
    user_dict, item_dict, word_dict = generate_dictionaries(args)  # user_dict={'id0': 0, 'id1': 1}, word_dict={',': 0}
    word_embeddings = generate_embeddings(args, word_dict)  # word_embedding = size(vocabulary)*300 matrix
    user_documents, item_documents, user_item_review_index = generate_documents(args, user_dict, item_dict)

    training_users, training_items, training_ratings = [], [], []
    validation_users, validation_items, validation_ratings = [], [], []
    test_users, test_items, test_ratings = [], [], []
    with open(os.path.join(args.data_directory, args.training_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            training_users.append(user_dict[line[0]])
            training_items.append(item_dict[line[1]])
            training_ratings.append(float(line[2]))
    with open(os.path.join(args.data_directory, args.validation_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            validation_users.append(user_dict[line[0]])
            validation_items.append(item_dict[line[1]])
            validation_ratings.append(float(line[2]))
    with open(os.path.join(args.data_directory, args.test_set), "r") as file_in:
        for line in file_in:
            line = line.split("\t\t")
            test_users.append(user_dict[line[0]])
            test_items.append(item_dict[line[1]])
            test_ratings.append(float(line[2]))
    user_indices, item_indices, user_scores, item_scores = generate_metadata(training_users, training_items, training_ratings, len(user_dict), len(item_dict))
    save(user_dict, "user_dict.dat", args.data_directory)
    save(item_dict, "item_dict.dat", args.data_directory)
    save(word_dict, "word_dict.dat", args.data_directory)
    save(word_embeddings, "word_embeddings.dat", args.data_directory)
    save(user_documents, "user_documents.dat", args.data_directory)
    save(item_documents, "item_documents.dat", args.data_directory)
    save(user_item_review_index, 'user_item_review_index.dat', args.data_directory)
    save(training_users, "training_users.dat", args.data_directory)
    save(training_items, "training_items.dat", args.data_directory)
    save(training_ratings, "training_ratings.dat", args.data_directory)
    save(validation_users, "validation_users.dat", args.data_directory)
    save(validation_items, "validation_items.dat", args.data_directory)
    save(validation_ratings, "validation_ratings.dat", args.data_directory)
    save(test_users, "test_users.dat", args.data_directory)
    save(test_items, "test_items.dat", args.data_directory)
    save(test_ratings, "test_ratings.dat", args.data_directory)
    save(user_indices, "user_indices.dat", args.data_directory)
    save(item_indices, "item_indices.dat", args.data_directory)
    save(user_scores, "user_scores.dat", args.data_directory)
    save(item_scores, "item_scores.dat", args.data_directory)


if __name__ == "__main__":
    config = get_config()
    preprocess(config)
