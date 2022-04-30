'''
@Author: Xiao Tong
@FileName: model.py
@CreateTime: 2022-04-30 20:01:50
@Description:

'''

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class CNN3D(torch.nn.Module):
    def __init__(self, out_channel, kernel_size_list, dropout):
        super(CNN3D, self).__init__()
        self.out_channel = out_channel
        self.kernel_size_list = kernel_size_list
        self.cnn_3d = torch.nn.Conv3d(3, out_channel, (3, 7, 7))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, image_input):
        convs_out = self.cnn_3d(image_input)
        pool_x = F.max_pool1d(convs_out.squeeze(-1), convs_out.size()[3])
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        return fc_x


class CNN3DLSTM(torch.nn.Module):
    def __init__(self, config, num_words, pretrained_word_embedding=None):
        super(CNN3DLSTM, self).__init__()
        self.embedding = EmbeddingLayer(num_words,
                                        config["model"]["word_embedding_dim"],
                                        pretrained_word_embedding)

        self.bi_rnn = BiRNNLayer(config["model"]["out_channel"],
                                 config["model"]["word_embedding_dim"],
                                 config["model"]["kernel_size"],
                                 config["model"]["dropout"])

        self.cnn3d = CNN3D(pretrained=True, progress=True)

        self.resnet_outdim = self.resnet.fc.out_features
        self.linear_out = torch.nn.Linear(self.resnet_outdim + config["model"]["kernel_size"] * config["model"]["out_channel"],
                                          config["data"]["num_classes"])

    def forward(self, image_input, input_x, segments):
        image_output = self.resnet(image_input)

        embedding_out = self.embedding(input_x)
        text_output = self.textcnn(embedding_out.unsqueeze(1))
        logits = self.linear_out(torch.cat([image_output, text_output], dim=-1))
        scores = torch.sigmoid(logits)

        start_idx, end_idx = 0, 0
        video_scores = []
        for video in segments:
            start_idx = end_idx
            end_idx += sum(video)
            video_scores.append(torch.max(scores[start_idx:end_idx], dim=0)[0])
        video_scores = torch.stack(video_scores, dim=0)
        return video_scores
