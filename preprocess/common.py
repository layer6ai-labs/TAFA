import argparse
import numpy as np
import os
# import spacy
import torch


def get_config():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data-directory", type=str, default=None)
    arg_parser.add_argument("--training-set", type=str, default=None)
    arg_parser.add_argument("--validation-set", type=str, default=None)
    arg_parser.add_argument("--test-set", type=str, default=None)
    arg_parser.add_argument("--training-split", type=float, default=None)
    arg_parser.add_argument("--validation-split", type=float, default=None)
    arg_parser.add_argument("--test-split", type=float, default=None)
    arg_parser.add_argument("--glove-filename", type=str, default="glove.840B.300d.txt")
    arg_parser.add_argument("--vocabulary-size", type=int, default=None)
    arg_parser.add_argument("--embedding-dim", type=int, default=None)
    arg_parser.add_argument("--word-threshold", type=int, default=None)
    arg_parser.add_argument("--document-threshold", type=int, default=None)
    arg_parser.add_argument("--random-seed", type=int, default=1905)
    return arg_parser.parse_args()


# def get_tokenizer():
    # return spacy.load("en")


def generate_embeddings(args, word_dict):
    word_embeddings = np.random.normal(loc=0.0, scale=0.01, size=(len(word_dict), args.embedding_dim))
    if args.glove_filename:
        with open(os.path.join(args.data_directory, args.glove_filename), "r") as file_in:
            for line in file_in:
                array = line.split()
                word = "".join(array[:-args.embedding_dim])
                if word in word_dict:
                    word_embeddings[word_dict[word]] = list(map(float, array[-args.embedding_dim:]))
    word_embeddings = torch.FloatTensor(word_embeddings)
    return word_embeddings


#  TODO fix rating non-unique problems

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


def tokenize(nlp, text):
    doc = nlp(text)
    return [token.text for token in doc]
