'''
@Author: Xiao Tong
@FileName: model.py
@CreateTime: 2022-04-30 20:01:50
@Description:

'''

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


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


class TextCNN(torch.nn.Module):
    def __init__(self, out_channel, word_embedding_dim, kernel_size_list, dropout):
        super(TextCNN, self).__init__()
        self.out_channel = out_channel
        self.word_embedding_dim = word_embedding_dim
        self.kernel_size_list = kernel_size_list
        self.convs = [torch.nn.Conv2d(1, out_channel, kernel_size=(w, word_embedding_dim)) for w in kernel_size_list]
        self.convs = torch.nn.ModuleList(self.convs)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_x):
        convs_out = [conv(input_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in convs_out]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        return fc_x


class TextCNNModel(torch.nn.Module):
    def __init__(self, config, num_words, pretrained_word_embedding=None):
        super(TextCNNModel, self).__init__()
        self.embedding = EmbeddingLayer(num_words,
                                        config["model"]["word_embedding_dim"],
                                        pretrained_word_embedding)

        self.textcnn = TextCNN(config["model"]["out_channel"],
                               config["model"]["word_embedding_dim"],
                               config["model"]["kernel_size"],
                               config["model"]["dropout"])

        self.linear_out = torch.nn.Linear(len(config["model"]["kernel_size"]) * config["model"]["out_channel"],
                                          config["data"]["num_classes"])

    def forward(self, input_x, segments):
        embedding_out = self.embedding(input_x)
        text_output = self.textcnn(embedding_out)
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
