'''
@Author: Xiao Tong
@FileName: model.py
@CreateTime: 2022-04-30 20:01:50
@Description:

'''

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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


class BiLSTMLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMLayer, self).__init__()
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


class LSTMModel(torch.nn.Module):
    def __init__(self, config, num_words, pretrained_word_embedding=None):
        super(LSTMModel, self).__init__()
        self.embedding = EmbeddingLayer(num_words,
                                        config["model"]["word_embedding_dim"],
                                        pretrained_word_embedding)

        self.bi_lstm = BiLSTMLayer(in_dim=config["model"]["word_embedding_dim"],
                                   hidden_dim=config["model"]["rnn_dim"],
                                   num_layers=config["model"]["rnn_num_layers"],
                                   dropout=config["model"]["dropout"])
        self.linear_out = torch.nn.Linear(2 * config["model"]["rnn_dim"], config["data"]["num_classes"])

    def forward(self, text_record, segments):
        input_x, lens = text_record 
        embedding_out = self.embedding(input_x)
        text_output = self.bi_lstm(embedding_out, lens)
        logits = self.linear_out(text_output)
        scores = torch.sigmoid(logits)

        start_idx, end_idx = 0, 0
        video_scores = []
        for video in segments:
            start_idx = end_idx
            end_idx += sum(video)
            video_scores.append(torch.max(scores[start_idx:end_idx], dim=0)[0])
        video_scores = torch.stack(video_scores, dim=0)
        return video_scores
