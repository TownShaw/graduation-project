'''
@Author: Xiao Tong
@FileName: harnn.py
@CreateTime: 2022-03-22 20:36:18
@Description:

'''

import torch
from harl import HARL
from torch.nn.utils.rnn import pad_packed_sequence


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, word_num, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(word_num, embedding_dim)

    def forward(self, input_x):
        return self.embedding(input_x)


class BiRNNLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(BiRNNLayer, self).__init__()
        self.bi_rnn = torch.nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, pack_x):
        seqs, lens = pad_packed_sequence(pack_x)
        output, _ = self.bi_rnn(seqs)
        avg_output = []
        for idx in range(len(lens)):
            avg = torch.mean(output[idx, :lens[idx], :])
            avg_output.append(avg)
        avg_output = torch.stack(avg_output, dim=0)
        return output, avg_output


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        return self.bce_logits_loss(logits, labels)


class HARNN(torch.nn.Module):
    def __init__(self, config, num_words, embedding_dim, rnn_dim, label_embedding_dim, num_classes_list, max_seq_len,
                 pretrained_word_embedding=None, pretrained_label_embedding=None):
        super(HARNN, self).__init__()
        self.embedding = EmbeddingLayer(word_num=num_words, embedding_dim=embedding_dim)
        self.bi_rnn = BiRNNLayer(embedding_dim, rnn_dim, num_layers=1, dropout=config.dropout)
        self.harl = HARL(rnn_dim=rnn_dim,
                         label_embedding_dim=label_embedding_dim,
                         num_classes_list=num_classes_list,
                         pretrained_label_embedding=pretrained_label_embedding)
        self.linear = torch.nn.Linear()
        self.loss = Loss()

    def forward(self, input_x):
        embedding_output = self.embedding(input_x)
        rnn_out, rnn_avg = self.bi_rnn(embedding_output)
        local_output_list, local_scores_list = self.harl(rnn_out)
        local_scores = torch.cat(local_scores_list, dim=-1)
        local_output = torch.stack(local_output_list, dim=0)
        global_output = self.linear(local_output)
        
        pass


if __name__ == "__main__":
    pass
