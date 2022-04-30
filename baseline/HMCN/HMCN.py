'''
@Author: Xiao Tong
@FileName: HMCN-F.py
@CreateTime: 2022-04-29 15:52:10
@Description:

'''

import torch
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, word_num, embedding_dim, pretrained_word_matrix=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(word_num, embedding_dim)
        if pretrained_word_matrix is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_word_matrix)
        else:
            xavier_normal_(self.embedding.weight.data)

    def forward(self, input_x):
        return self.embedding(input_x)


class BiRNNLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(BiRNNLayer, self).__init__()
        self.bi_rnn = torch.nn.LSTM(in_dim,
                                    hidden_dim,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    batch_first=True,
                                    bidirectional=True)

    def forward(self, input_x, lens):
        text_record = pack_padded_sequence(input_x, lens, batch_first=True, enforce_sorted=False)
        output, _ = self.bi_rnn(text_record)
        output, lens = pad_packed_sequence(output, batch_first=True)
        avg_output = []
        lens = lens.tolist()
        for idx in range(len(lens)):
            avg = torch.mean(output[idx, :lens[idx], :], dim=0)
            avg_output.append(avg)
        avg_output = torch.stack(avg_output, dim=0)
        return avg_output


class HMCNLoss(torch.nn.Module):
    def __init__(self):
        super(HMCNLoss, self).__init__()
        self.bceloss = torch.nn.BCELoss()

    def forward(self, scores, labels):
        return self.bceloss(scores, labels)


class ForwardUnit(torch.nn.Module):
    def __init__(self, in_channel, hidden_out_channel, feature_out_channel, num_classes):
        super(ForwardUnit, self).__init__()
        self.linear1 = torch.nn.Linear(in_channel, hidden_out_channel)
        self.linear2 = torch.nn.Linear(hidden_out_channel, feature_out_channel)
        self.linear3 = torch.nn.Linear(feature_out_channel, num_classes)

    def forward(self, input_x, last_h, local_scores_list: list):
        unit_input = torch.cat([input_x, last_h], dim=-1)
        hidden_out = torch.relu(self.linear1(unit_input))
        local_feature = torch.relu(self.linear2(hidden_out))
        local_scores = torch.sigmoid(self.linear3(local_feature))
        local_scores_list.append(local_scores)
        return input_x, hidden_out, local_scores_list


class HMCN_F(torch.nn.Module):
    def __init__(self, config: dict):
        super(HMCN_F, self).__init__()
        self.alpha = config["HMCN-F"]["alpha"]
        self.rnn_dim = config["HMCN-F"]["rnn_dim"]
        self.hidden_dim = config["HMCN-F"]["hidden_dim"]
        self.feature_dim = config["HMCN-F"]["feature_dim"]
        self.num_classes = config["data"]["num_classes"]
        self.num_classes_list = config["data"]["num_classes_list"]
        self.network = []
        for idx in range(len(self.num_classes_list)):
            if idx == 0:
                self.network.append(ForwardUnit(2 * self.rnn_dim, self.hidden_dim, self.feature_dim, self.num_classes_list[idx]))
            else:
                self.network.append(ForwardUnit(2 * self.rnn_dim + self.hidden_dim, self.hidden_dim, self.feature_dim, self.num_classes_list[idx]))
        self.network = mySequential(*self.network)
        self.linear = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, input_x):
        local_scores_list = []
        _, global_feature, local_scores_list = self.network(input_x,
                                                            torch.FloatTensor().to(input_x.device),
                                                            local_scores_list)
        local_scores = torch.cat(local_scores_list, dim=-1)
        global_scores = torch.sigmoid(self.linear(global_feature))
        final_scores = (1 - self.alpha) * local_scores + self.alpha * global_scores
        return final_scores


class RecurrentUnit(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(RecurrentUnit, self).__init__()
        self.linear_F = torch.nn.Linear(in_channel, hidden_channel)
        self.linear_I = torch.nn.Linear(in_channel, hidden_channel)
        self.linear_C = torch.nn.Linear(in_channel, hidden_channel)
        self.linear_O = torch.nn.Linear(in_channel, hidden_channel)

    def forward(self, input_x, last_h, last_C):
        unit_input = torch.cat([input_x, last_h], dim=-1)
        F = torch.sigmoid(self.linear_F(unit_input))
        I = torch.sigmoid(self.linear_I(unit_input))
        O = torch.sigmoid(self.linear_O(unit_input))
        C_hat = torch.tanh(self.linear_C(unit_input))
        curr_C = torch.multiply(F, last_C) + torch.multiply(I, C_hat)
        hidden_state = torch.multiply(torch.tanh(curr_C), O)
        return input_x, hidden_state, curr_C


class HMCN_R(torch.nn.Module):
    def __init__(self, config: dict):
        super(HMCN_R, self).__init__()
        self.alpha = config["HMCN-R"]["alpha"]
        self.batch_size = config["HMCN-R"]["batch_size"]
        self.max_seq_len = config["HMCN-R"]["max_seq_len"]
        self.rnn_dim = config["HMCN-R"]["rnn_dim"]
        self.hidden_dim = config["HMCN-R"]["hidden_dim"]
        self.num_classes = config["data"]["num_classes"]
        self.num_classes_list = config["data"]["num_classes_list"]

        self.recurrent = RecurrentUnit(2 * self.rnn_dim + self.hidden_dim, self.hidden_dim)
        self.global_out = torch.nn.Linear(2 * self.rnn_dim + self.hidden_dim, self.num_classes)
        self.local_out_nets = [torch.nn.Linear(self.hidden_dim, self.num_classes_list[idx]) for idx in range(len(self.num_classes_list))]

    def forward(self, input_x):
        self.local_out_nets = [net.to(input_x.device) for net in self.local_out_nets]
        local_scores_list = []

        hidden_state = xavier_normal_(torch.rand(input_x.shape[0], self.hidden_dim, dtype=torch.float32)).to(input_x.device)
        memory_cell = xavier_normal_(torch.rand(input_x.shape[0], self.hidden_dim, dtype=torch.float32)).to(input_x.device)
        for level in range(len(self.num_classes_list)):
            input_x, hidden_state, memory_cell = self.recurrent(input_x, hidden_state, memory_cell)
            scores = torch.sigmoid(self.local_out_nets[level](hidden_state))
            local_scores_list.append(scores)
        local_scores = torch.cat(local_scores_list, dim=-1)
        global_scores = torch.sigmoid(self.global_out(torch.cat([input_x, hidden_state], dim=-1)))
        final_scores = (1 - self.alpha) * local_scores + self.alpha * global_scores
        return final_scores


class HMCN(torch.nn.Module):
    def __init__(self, config, model_name, num_words, pretrained_word_embedding=None):
        super(HMCN, self).__init__()
        MODELS = {
            "HMCN-F": HMCN_F,
            "HMCN-R": HMCN_R
        }

        self.embedding = EmbeddingLayer(word_num=num_words,
                                        embedding_dim=config[model_name]["word_embedding_dim"],
                                        pretrained_word_matrix=pretrained_word_embedding)

        self.bi_rnn = BiRNNLayer(in_dim=config[model_name]["word_embedding_dim"],
                                 hidden_dim=config[model_name]["rnn_dim"],
                                 num_layers=config[model_name]["rnn_num_layers"],
                                 dropout=config[model_name]["dropout"])
        self.hmcn = MODELS[model_name](config)

    def forward(self, text_record, segments):
        text_pad, lens = text_record
        embedding_output = self.embedding(text_pad)
        rnn_avg = self.bi_rnn(embedding_output, lens)
        final_scores = self.hmcn(rnn_avg)

        start_idx, end_idx = 0, 0
        video_final_scores = []
        for video in segments:
            start_idx = end_idx
            end_idx += sum(video)
            video_final_scores.append(torch.max(final_scores[start_idx:end_idx], dim=0)[0])
        video_final_scores = torch.stack(video_final_scores, dim=0)
        return video_final_scores
