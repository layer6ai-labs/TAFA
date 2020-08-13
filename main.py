from utility.argument_check import *
import torch.nn as nn
import git
from models.TAFA import tafa
from utility.data import *
from scipy.sparse import csr_matrix
from utility.model_helper import binarize_dataset, convert_to_rating_matrix_from_lists
from utility.progress import WorkSplitter
from utility.predictor import predict
from utility.metrics import evaluate


def main(args):
    progress = WorkSplitter()
    args_dictionary = vars(args)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print('current git hash is {0}'.format(sha))
    print('loading data directory: {0}'.format(args.data_directory))
    print("Algorithm: {0}".format(args.model))
    progress.section("Loading Data")
    if 'yelp' in args.data_directory:
        train_users, train_items, train_ratings, val_users, val_items, val_ratings, user_documents, item_documents, word_dict, word_embeddings = load_dataset_yelp(args)
    elif 'amazon' in args.data_directory:
        train_users, train_items, train_ratings, val_users, val_items, val_ratings, user_documents, item_documents, word_dict, word_embeddings = load_dataset_amazon(args)
    else:
        raise NotImplementedError

    if args.one_class:
        train_users, train_items, train_ratings = binarize_dataset(args.one_class_threshold, train_users, train_items,
                                                                   train_ratings)
        val_users, val_items, val_ratings = binarize_dataset(args.one_class_threshold, val_users, val_items,
                                                             val_ratings)

    train = (train_users, train_items, train_ratings)
    validation = (val_users, val_items, val_ratings)
    document_data = (user_documents, item_documents, word_dict, word_embeddings)

    args_dictionary['train'] = train
    args_dictionary['val'] = validation
    args_dictionary['document_data'] = document_data
    training_result = tafa(**args_dictionary)

    progress.section("Predict")
    # generate train and validation matrices
    num_users = len(user_documents)
    num_items = len(item_documents)
    matrix_train = convert_to_rating_matrix_from_lists(num_users, num_items, train_users, train_items,
                                                       train_ratings, True)

    matrix_val = convert_to_rating_matrix_from_lists(num_users, num_items, val_users, val_items, val_ratings, True)
    matrix_train_csr = csr_matrix(matrix_train)
    matrix_val_csr = csr_matrix(matrix_val)

    prediction = predict(training_result['best_prediction'], None, None, args.top_k, matrix_train_csr)
    # np.save('{2}/R_{0}_{1}.npy'.format(args.model, args.rank, 'latent'), prediction)
    progress.subsection("Evaluation")
    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
    result = evaluate(prediction, matrix_val_csr, metric_names, [5, 10, 15, 20, 30, 40, 50])
    for metric in result.keys():
        print("{0}:{1}".format(metric, result[metric]))
    print('best iteration is {0}'.format(training_result['best_iteration']))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser('main function')
    parser.add_argument('--data_directory', type=str, default="/home/joey/Documents/tafa/amazon_music/")
    parser.add_argument('--model', type=str, default="rnn_nceautorec_ee")
    parser.add_argument('--iteration', type=check_int_positive, default=200)
    parser.add_argument('--lam', type=check_float_positive, default=1)
    parser.add_argument('--rank', type=check_int_positive, default=500)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--rec_learning_rate', type=check_float_positive, default=1e-4)
    parser.add_argument('--lang_learning_rate', type=check_float_positive, default=1e-4)
    parser.add_argument('--glove_embedding_size', type=check_int_positive, default=300)
    parser.add_argument('--elmo_embedding_size', type=int, default=None)
    parser.add_argument('--encoder_hidden_size', type=check_int_positive, default=64)
    parser.add_argument('--attention_size', type=check_int_positive, default=256)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--encoder_num_layers', type=check_int_positive, default=1)
    parser.add_argument('--encoder_dropout_rate', type=float, default=0.0)
    parser.add_argument('--encoder_rnn_type', default=nn.LSTM)
    parser.add_argument('--encoder_concat_layers', type=int, default=0)
    parser.add_argument('--attention_hidden_size', type=check_int_positive, default=256)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.0)
    parser.add_argument('--mc_times', type=int, default=5)
    parser.add_argument('--separate', type=int, default=1)
    parser.add_argument('--decoder_hidden_size', type=check_int_positive, default=64)
    parser.add_argument('--decoder_dropout_rate', type=float, default=0.0)
    parser.add_argument('--decoder_loss', default=nn.CrossEntropyLoss(reduction='none'))
    parser.add_argument('--feature_mask', type=int, default=0)
    parser.add_argument('--custom_mask', type=int, default=0)
    parser.add_argument('--num_heads', type=check_int_positive, default=2)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--one_class', type=int, default=1)
    parser.add_argument('--one_class_threshold', type=int, default=3)  # binarize threshold
    parser.add_argument('--threshold', type=int, default=-1)  # for nce
    parser.add_argument('--root', type=check_float_positive, default=1)
    parser.add_argument('--mode', type=str, default='joint')
    parser.add_argument('--nce_loss_positive_only', type=int, default=0)
    parser.add_argument('--predict_loss_positive_only', type=int, default=0)
    parser.add_argument('--sample_strategy', type=str, default='random_batch')
    parser.add_argument('--distance_dropout_prob', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--cml_embedding_dim', type=check_int_positive, default=5)
    parser.add_argument('--norm_factor', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_len', type=check_int_positive, default=302)
    parser.add_argument('--rec_batch_size', type=int, default=100)
    parser.add_argument('--lang_feature_batch_size', type=int, default=32)
    parser.add_argument('--max_lang_iterations', type=int, default=128)
    parser.add_argument('--gradient_clipping', type=float, default=10.0)
    parser.add_argument('--rec_epoch', type=int, default=-1)
    parser.add_argument('--fix_encoder', type=int, default=0)
    parser.add_argument('--criteria', type=str, default='NDCG')
    parser.add_argument('--top_k', type=check_int_positive, default=50)
    args = parser.parse_args()
    main(args)
