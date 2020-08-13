import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from utility.progress import WorkSplitter
from utility.model_helper import generate_nce_matrix, get_optimizer, calculate_mse, generate_batches_ids, convert_to_rating_matrix_from_lists
from scipy.sparse import csr_matrix
import time
from utility.predictor import predict
from utility.metrics import evaluate
import copy
import math

np.random.seed(47)
random.seed(47)
torch.random.manual_seed(47)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, rnn_type, concat_layer):
        super(BidirectionalRNN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.concat_layers = concat_layer
        self.rnn_list = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size * 2
            self.rnn_list.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        lengths = x_mask.data.eq(0).sum(1)
        _, index_sort = torch.sort(lengths, dim=0, descending=True)
        _, index_unsort = torch.sort(index_sort, dim=0, descending=False)
        lengths = list(lengths[index_sort])
        x = x.index_select(dim=0, index=index_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0.0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnn_list[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        if self.concat_layers:
            output = torch.cat(outputs[1:], dim=2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(dim=0, index=index_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(
                output.data.type())
            output = torch.cat([output, padding], 1)
        # indices = x_mask.shape[1] - x_mask.sum(dim=1) - 1
        # output = output[torch.arange(output.shape[0]), indices, :]
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0):
        super(SelfAttention, self).__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, input_encoded, input_mask):

        scores = F.dropout(F.tanh(self.linear1(input_encoded)), p=self.dropout_rate, training=self.training)
        scores = self.linear2(scores).squeeze(2)
        scores.data.masked_fill_(input_mask.data, -float('inf'))

        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(dim=1), input_encoded).squeeze(dim=1)

        return context


class SelfAttentionFusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_topics, dropout_rate=0):
        super(SelfAttentionFusion, self).__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

        self.linear_x = nn.Linear(num_topics, hidden_size)

    def forward(self, input_encoded, input_mask, x):

        input_x = self.linear_x(x).unsqueeze(1)

        input_x = input_x.expand_as(input_encoded)

        input_concat = torch.cat((input_encoded, input_x), dim=2)

        scores = F.dropout(F.tanh(self.linear1(input_concat)), p=self.dropout_rate, training=self.training)
        scores = self.linear2(scores).squeeze(2)
        scores.data.masked_fill_(input_mask.data, -float('inf'))

        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(dim=1), input_encoded).squeeze(dim=1)

        return context


class Encoder(nn.Module):
    def __init__(self, vocabulary_size, glove_embedding_size, encoder_hidden_size,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics):
        super(Encoder, self).__init__()

        hidden_size = encoder_hidden_size * 2

        self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=glove_embedding_size)
        self.word_encoder = BidirectionalRNN(
            input_size=glove_embedding_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers
        )
        self.word_attention = SelfAttention(input_size=hidden_size, hidden_size=hidden_size)

        self.sent_encoder = BidirectionalRNN(
            input_size=hidden_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers
        )
        self.sent_attention = SelfAttentionFusion(hidden_size, hidden_size, num_topics)

    def forward(self, word_indices, word_mask, sent_lengths, x):

        word_embedded = self.word_embeddings(word_indices)
        word_encoded = self.word_encoder.forward(x=word_embedded, x_mask=word_mask)
        word_encoded = self.word_attention(input_encoded=word_encoded, input_mask=word_mask.bool())

        sent_input, sent_mask = _align_sent(word_encoded, sent_lengths)

        sent_encoded = self.sent_encoder.forward(x=sent_input, x_mask=sent_mask)
        sent_encoded = self.sent_attention(input_encoded=sent_encoded, input_mask=sent_mask.bool(), x=x)

        return sent_encoded

    def initialize(self, word_embeddings):
        self.word_embeddings.weight.data.copy_(word_embeddings)


def _align_sent(batch_input, sent_lenghts, sent_max=None):

    hidden_dim = batch_input.size(-1)
    passage_num = len(sent_lenghts)

    if sent_max is not None:
        max_len = sent_max
    else:
        max_len = np.max(sent_lenghts)

    sent_input = torch.zeros(passage_num, max_len, hidden_dim).cuda()
    sent_mask = torch.ones(passage_num, max_len).cuda()

    init_index = 0

    for index, length in enumerate(sent_lenghts):
        end_index = init_index + length

        temp_input = batch_input[init_index:end_index, :]

        if temp_input.size(0) > max_len:
            temp_input = temp_input[:max_len]

        sent_input[index, :length, :] = temp_input
        sent_mask[index, :length] = 0

        init_index = end_index

    return sent_input, sent_mask


class EncoderHAttn(nn.Module):
    def __init__(self, vocabulary_size, glove_embedding_size, encoder_hidden_size,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics):
        super(EncoderHAttn, self).__init__()

        self.word_embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=glove_embedding_size)
        self.context_encoder = BidirectionalRNN(
            input_size=glove_embedding_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout_rate=encoder_dropout_rate,
            rnn_type=encoder_rnn_type,
            concat_layer=encoder_concat_layers)
        self.fc1 = nn.Linear(2 * encoder_hidden_size, encoder_hidden_size * 2)

    def forward(self, word_indices, word_mask, x):
        word_embedded = self.word_embeddings(word_indices)
        context_encoded = self.context_encoder.forward(x=word_embedded, x_mask=word_mask)
        context_encoded = self.fc1(context_encoded)
        return context_encoded

    def initialize(self, word_embeddings):
        self.word_embeddings.weight.data.copy_(word_embeddings)


class ModalityFusion(nn.Module):
    def __init__(self, md1_dim, md2_dim, attention_size=128):
        super(ModalityFusion, self).__init__()
        self.md1_in = nn.Linear(md1_dim, attention_size)
        self.md2_in = nn.Linear(md2_dim, attention_size)

        self.linear_k = nn.Linear(attention_size, 1)
        self.linear_q = nn.Linear(attention_size, attention_size)

    def forward(self, md1_vec, md2_vec):
        batch_size = md1_vec.size(0)
        md1_dim = md1_vec.size(1)
        md2_dim = md2_vec.size(1)

        md1_out = self.md1_in(md1_vec)
        md2_out = self.md2_in(md2_vec)

        md1_out_attn = self.linear_k(torch.tanh(md1_out))
        md2_out_attn = self.linear_k(torch.tanh(md2_out))

        attn_scores = torch.cat((md1_out_attn, md2_out_attn), dim=1)
        attn_scores = F.softmax(attn_scores, -1).unsqueeze(1)

        # md1_vec = self.linear_q(torch.tanh(md1_out))
        # md2_vec = self.linear_q(torch.tanh(md2_out))

        md1_vec = torch.tanh(self.linear_q(md1_out))
        md2_vec = torch.tanh(self.linear_q(md2_out))

        md_vecs = torch.stack((md1_vec, md2_vec), dim=1)

        out_vec = torch.bmm(attn_scores, md_vecs).squeeze()

        return out_vec


class NCEAutoRecNLP(nn.Module):
    def __init__(self, num_users, num_items, num_topics, user, vocabulary_size, glove_embedding_size, encoder_hidden_size, attention_size, dropout_p,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, word_embeddings,
                 activation='relu', loss='mse', nce_head_positive_only=1, predict_head_positive_only=1):

        super(NCEAutoRecNLP, self).__init__()
        self.nce_head_positive_only = nce_head_positive_only
        self.predict_head_positive_only = predict_head_positive_only
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        if loss == 'mse':
            self.loss = nn.MSELoss()
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if user:
            self.encode = nn.Linear(num_items, num_topics)
            # self.nce_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_items)
            # self.predict_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_items)

            self.nce_decode = nn.Linear(attention_size, num_items)
            self.predict_decode = nn.Linear(attention_size, num_items)

            self.text_encoder = Encoder(vocabulary_size, glove_embedding_size, encoder_hidden_size,
                                    encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers,
                                    num_topics)

            self.modality_fusion = ModalityFusion(num_topics, encoder_hidden_size * 2,
                                                  attention_size=attention_size)

        else:
            self.encode = nn.Linear(num_users, num_topics)
            # self.nce_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_users)
            # self.predict_decode = nn.Linear(num_topics + encoder_hidden_size * 2, num_users)

            self.nce_decode = nn.Linear(attention_size, num_users)
            self.predict_decode = nn.Linear(attention_size, num_users)

            self.text_encoder = Encoder(vocabulary_size, glove_embedding_size, encoder_hidden_size,
                                    encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers,
                                    num_topics)

            self.modality_fusion = ModalityFusion(num_topics, encoder_hidden_size * 2,
                                                  attention_size=attention_size)

        self.text_encoder.initialize(word_embeddings)

    def get_feature(self, rating_matrix):
        return self.encode(rating_matrix)

    def forward(self, rating_matrix):
        x = self.encode(rating_matrix)
        x = self.activation(x)
        out = self.predict_decode(x)
        return out

    def forward_nce(self, rating_matrix):
        x = self.encode(rating_matrix)
        x = self.activation(x)
        out = self.nce_decode(x)
        return out

    def forward_two_heads(self, rating_matrix):
        x = self.encode(rating_matrix)
        x = self.activation(x)
        nce_out = self.nce_decode(x)
        predict_out = self.predict_decode(x)
        return nce_out, predict_out

    def forward_two_heads_language(self, rating_matrix_batch, language_inputs_batch):

        x = F.dropout(rating_matrix_batch, p=self.dropout_p, training=self.training)
        x = self.encode(x)
        x = self.activation(x)

        [_, word_indices, word_mask, _, _, length_lists] = language_inputs_batch

        textual_features = self.text_encoder.forward(word_indices=word_indices, word_mask=word_mask, sent_lengths=length_lists, x=x)

        # predicted_features = torch.split(textual_features, length_lists)
        # mean_predicted_features = torch.stack([torch.mean(vec, dim=0) for vec in predicted_features], dim=0).cuda()

        modality_fusion = self.modality_fusion.forward(x, textual_features)

        # all_prediction = torch.cat((x, mean_predicted_features), 1)
        all_prediction = modality_fusion

        predict_out = self.predict_decode(all_prediction)
        nce_out = self.nce_decode(all_prediction)

        return nce_out, predict_out

    def forward_language(self, rating_matrix_batch, language_inputs_batch):
        x = self.encode(rating_matrix_batch)
        x = self.activation(x)

        [_, word_indices, word_mask, _, _, length_lists] = language_inputs_batch

        textual_features = self.text_encoder.forward(word_indices=word_indices, word_mask=word_mask, sent_lengths=length_lists, x=x)

        # predicted_features = torch.split(textual_features, length_lists)
        # mean_predicted_features = torch.stack([torch.mean(vec, dim=0) for vec in predicted_features], dim=0).cuda()

        modality_fusion = self.modality_fusion.forward(x, textual_features)

        # all_prediction = torch.cat((x, mean_predicted_features), 1)
        all_prediction = modality_fusion

        out = self.predict_decode(all_prediction)

        return out


class RNN_NCEAutoRec(object):
    def __init__(self, vocabulary_size, glove_embedding_size, elmo_embedding_size, encoder_hidden_size, attention_size, dropout_p,
                 encoder_num_layers, encoder_dropout_rate, encoder_rnn_type, encoder_concat_layers, num_topics, max_len,
                 mc_times, separate, decoder_hidden_size, decoder_dropout_rate, word_dict, decoder_loss, num_users,
                 num_items, activation, autoencoder_loss,
                 nce_head_positive_only, predict_head_positive_only, word_embeddings):

        self.user_autoencoder = NCEAutoRecNLP(num_users, num_items, num_topics, True, vocabulary_size, glove_embedding_size,
                                              encoder_hidden_size, attention_size, dropout_p, encoder_num_layers, encoder_dropout_rate, encoder_rnn_type,
                                              encoder_concat_layers, word_embeddings, activation, autoencoder_loss,
                                           nce_head_positive_only, predict_head_positive_only).cuda()

        self.item_autoencoder = NCEAutoRecNLP(num_users, num_items, num_topics, False, vocabulary_size, glove_embedding_size,
                                              encoder_hidden_size, attention_size, dropout_p, encoder_num_layers, encoder_dropout_rate, encoder_rnn_type,
                                              encoder_concat_layers, word_embeddings, activation, autoencoder_loss,
                                           nce_head_positive_only, predict_head_positive_only).cuda()

        self.max_len = max_len
        self.word_dict = word_dict

    @staticmethod
    def update_nce_autorec(model, max_autoencoder_iteration, lam, matrix, nce_matrix, mode, predict_optimizer,
                           step, documents, num_entries, autoencoder_batch_size, word_dict, max_len):

        t1 = time.time()
        model.train()
        current_step = 0
        if max_autoencoder_iteration == 0:
            t2 = time.time()
            print(
                'finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                  mode))
            return step

        if mode == 'joint':
            for i in range(max_autoencoder_iteration):

                indices = list(range(num_entries))
                random.shuffle(indices)

                for start_index in range(0, len(indices), autoencoder_batch_size):
                    end_index = min(start_index + autoencoder_batch_size, len(indices))

                    ids_batch = indices[start_index:end_index]

                    user_matrix_batch = matrix[ids_batch]
                    user_nce_batch = nce_matrix[ids_batch]

                    raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists = \
                        generate_batches_ids(documents, word_dict, max_len, ids_batch)

                    nce_out, predict_out = model.forward_two_heads_language(user_matrix_batch, [raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists])

                    reg_loss = (model.encode.weight ** 2).mean() * lam + (model.nce_decode.weight ** 2).mean() * lam + (
                            model.predict_decode.weight ** 2).mean() * lam
                    nce_loss = calculate_mse(nce_out, user_nce_batch, model.nce_head_positive_only)
                    predict_loss = calculate_mse(predict_out, user_matrix_batch, model.predict_head_positive_only)

                    loss = nce_loss + predict_loss + reg_loss

                    predict_optimizer.zero_grad()
                    loss.backward()
                    predict_optimizer.step()
                    current_step += 1

        t2 = time.time()
        print('finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                mode))
        return current_step + step

    def update_nce_autorec_joint(self, user_autoencoder, item_autoencoder, max_autoencoder_iteration, lam, user_matrix,
                                 user_nce_matrix, item_matrix, item_nce_matrix, mode, predict_optimizer, step,
                                 user_documents, item_documents, num_users, num_items, autoencoder_batch_size,
                                 word_dict, max_len):

        t1 = time.time()
        user_autoencoder.train()
        item_autoencoder.train()

        current_step = 0

        if mode == 'joint':
            for i in range(max_autoencoder_iteration):

                user_indices = np.random.randint(0, num_users, autoencoder_batch_size).tolist()

                user_matrix_batch = user_matrix[user_indices].cuda()
                user_nce_batch = user_nce_matrix[user_indices].cuda()

                user_raw_texts, user_word_indices, user_word_mask, user_index_pointers, user_user_indices, user_length_lists\
                    = generate_batches_ids(user_documents, word_dict, max_len, user_indices)

                user_nce_out, user_predict_out = user_autoencoder.forward_two_heads_language(user_matrix_batch,
                                                                                             [user_raw_texts,
                                                                                              user_word_indices,
                                                                                              user_word_mask,
                                                                                              user_index_pointers,
                                                                                              user_user_indices,
                                                                                              user_length_lists])

                user_reg_loss = (user_autoencoder.encode.weight ** 2).mean() * lam + \
                                (user_autoencoder.nce_decode.weight ** 2).mean() * lam + \
                                (user_autoencoder.predict_decode.weight ** 2).mean() * lam

                user_nce_loss = calculate_mse(user_nce_out, user_nce_batch, user_autoencoder.nce_head_positive_only)

                user_predict_loss = calculate_mse(user_predict_out, user_matrix_batch,
                                                  user_autoencoder.predict_head_positive_only)

                user_loss = user_nce_loss + user_predict_loss + user_reg_loss

                loss = user_loss

                predict_optimizer.zero_grad()
                loss.backward()
                predict_optimizer.step()
                current_step += 1

        t2 = time.time()
        print('finished updating nceautorec with mode {2} with {0} steps in {1} seconds'.format(current_step, t2 - t1,
                                                                                                mode))
        return current_step + step

    def inference_nce_autorec(self, model, matrix, nce_matrix, documents, num_entries, autoencoder_batch_size, word_dict, max_len):

        indices = list(range(num_entries))
        predict_out_list = []

        with torch.no_grad():
            model.eval()
            for start_index in range(0, len(indices), autoencoder_batch_size):
                end_index = min(start_index + autoencoder_batch_size, len(indices))

                ids_batch = indices[start_index:end_index]

                user_matrix_batch = matrix[ids_batch].cuda()
                # user_nce_batch = nce_matrix[ids_batch]

                raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists = \
                    generate_batches_ids(documents, word_dict, max_len, ids_batch)

                predict_out = model.forward_language(user_matrix_batch, [raw_texts, word_indices, word_mask, index_pointers, user_indices, length_lists]).cpu()
                predict_out_list.extend(predict_out)

        model.train()
        return torch.stack(predict_out_list)

    @staticmethod
    def evaluate_model(prediction, matrix_train_csr, train_data, matrix_val_csr, val_data):
        prediction_topK = predict(prediction, None, None, 50, matrix_train_csr)
        result = evaluate(prediction_topK, matrix_val_csr, ['R-Precision', 'NDCG', 'Precision', 'Recall'], [50])
        # training_users, training_items, training_ratings = train_data
        # validation_users, validation_items, validation_ratings = val_data
        # training_mse = evaluate_mse(prediction, training_users, training_items, training_ratings)
        # validation_mse = evaluate_mse(prediction, validation_users, validation_items, validation_ratings)
        # print(training_mse, validation_mse)
        # result['train_mse'] = [training_mse.item(), 0]
        # result['val_mse'] = [validation_mse.item(), 0]
        return result

    def train_model(self, train, val, user_documents, item_documents, iteration, lam, root, threshold, optimizer, momentum,
                    weight_decay, rec_learning_rate, autoencoder_batch_size, autoencoder_epoch,
                    mode, criteria, word_dict, max_len):

        autoencoder_decoder_step = 0

        (train_users, train_items, train_ratings) = train
        (val_users, val_items, val_ratings) = val
        num_users = len(user_documents)
        num_items = len(item_documents)
        matrix_train = convert_to_rating_matrix_from_lists(num_users, num_items, train_users, train_items,
                                                                 train_ratings, True)

        matrix_val = convert_to_rating_matrix_from_lists(num_users, num_items, val_users, val_items, val_ratings, True)
        matrix_train_csr = csr_matrix(matrix_train)
        matrix_val_csr = csr_matrix(matrix_val)

        user_ratings = torch.from_numpy(matrix_train).float()
        user_nce_matrix = torch.from_numpy(generate_nce_matrix(matrix_train, root, threshold)).float()

        item_ratings = torch.from_numpy(matrix_train.T).float()
        item_nce_matrix = torch.from_numpy(generate_nce_matrix(matrix_train.T, root, threshold)).float()

        if 'mse' in criteria:
            best_dict = {criteria: [float('inf'), 0]}
        else:
            best_dict = {criteria: [-1, 0]}

        num_users, num_items = matrix_train.shape

        predict_optimizer = get_optimizer(optimizer, [self.user_autoencoder, self.item_autoencoder], rec_learning_rate,
                                          momentum,
                                          weight_decay)

        for i in range(iteration):

            autoencoder_decoder_step = self.update_nce_autorec_joint\
                    (self.user_autoencoder, self.item_autoencoder, autoencoder_epoch, lam, user_ratings, user_nce_matrix,
                     item_ratings, item_nce_matrix, mode, predict_optimizer, autoencoder_decoder_step, user_documents, item_documents,
                     num_users, num_items, autoencoder_batch_size, word_dict, max_len)

            user_inference = self.inference_nce_autorec(self.user_autoencoder, user_ratings, user_nce_matrix, user_documents,
                                                        num_users, autoencoder_batch_size, word_dict, max_len)

            prediction = user_inference
            prediction_numpy = prediction.detach().cpu().numpy()
            result = self.evaluate_model(prediction_numpy, matrix_train_csr, train, matrix_val_csr, val)
            if 'mse' in criteria and result[criteria][0] < best_dict[criteria][0]:
                best_dict = copy.deepcopy(result)
                best_dict['best_iteration'] = i + 1
                best_dict['best_prediction' ] = copy.deepcopy(prediction_numpy)
            elif 'NDCG' in criteria and result[criteria][0] > best_dict[criteria][0]:
                best_dict = copy.deepcopy(result)
                best_dict['best_iteration'] = i + 1
                best_dict['best_prediction'] = copy.deepcopy(prediction_numpy)
            print('current iteration is {0}'.format(i + 1))
            print('current {0} is {1}'.format(criteria, result[criteria]))
            print('best iteration so far is {0}'.format(best_dict['best_iteration']))
            print('best {0} is {1}'.format(criteria, best_dict[criteria]))
        return best_dict


def tafa(train, val, document_data, iteration=15, lam=100, rank=100,
         optimizer='Adam', threshold=-1, root=1.1, mode='joint',
         rec_learning_rate=1e-4, activation_function='relu',
         loss_function='mse', nce_loss_positive_only=0, predict_loss_positive_only=0, momentum=0, weight_decay=0,
         glove_embedding_size=300, elmo_embedding_size=None, encoder_hidden_size=64, attention_size=256, dropout_p=0.5, encoder_num_layers=1,
         encoder_dropout_rate=0, encoder_rnn_type=nn.LSTM, encoder_concat_layers=False, max_len=302,
         mc_times=5, separate=1, decoder_hidden_size=64, decoder_dropout_rate=0,
         decoder_loss=nn.CrossEntropyLoss(reduction='none'), rec_batch_size=100,
         rec_epoch=1, criteria='NDCG', **unused):
    progress = WorkSplitter()
    (user_documents, item_documents, word_dict, word_embeddings) = document_data
    num_users = len(user_documents)
    num_items = len(item_documents)
    if rec_epoch == -1:
        rec_epoch = math.ceil(num_users / rec_batch_size)

    model = RNN_NCEAutoRec(vocabulary_size=len(word_dict), glove_embedding_size=glove_embedding_size, elmo_embedding_size=elmo_embedding_size,
                           encoder_hidden_size=encoder_hidden_size, attention_size=attention_size, dropout_p=dropout_p, encoder_num_layers=encoder_num_layers,
                           encoder_dropout_rate=encoder_dropout_rate, encoder_rnn_type=encoder_rnn_type,
                           encoder_concat_layers=encoder_concat_layers, num_topics=rank, max_len=max_len,
                           mc_times=mc_times, separate=separate, decoder_hidden_size=decoder_hidden_size,
                           decoder_dropout_rate=decoder_dropout_rate, word_dict=word_dict, decoder_loss=decoder_loss,
                           num_users=num_users, num_items=num_items, activation=activation_function,
                           autoencoder_loss=loss_function, nce_head_positive_only=nce_loss_positive_only,
                           predict_head_positive_only=predict_loss_positive_only, word_embeddings=word_embeddings)
    result = model.train_model(train, val, user_documents, item_documents, iteration, lam, root, threshold,
                               optimizer, momentum, weight_decay, rec_learning_rate,
                               rec_batch_size, rec_epoch, mode, criteria,
                               word_dict, max_len)
    return result
