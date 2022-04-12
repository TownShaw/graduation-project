'''
@Author: Xiao Tong
@FileName: harnn.py
@CreateTime: 2022-03-22 20:36:18
@Description:

'''

import torch
from harl import HARL
from torch.nn.init import xavier_normal_
from torchvision.models import resnet34
from torch.nn.utils.rnn import pad_packed_sequence


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, word_num, embedding_dim, pretrained_word_matrix=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(word_num, embedding_dim)
        if pretrained_word_matrix:
            self.embedding.weight.data = torch.from_numpy(pretrained_word_matrix)
        else:
            xavier_normal_(self.embedding.weight.data)

    def forward(self, input_x):
        return self.embedding(input_x)


class BiRNNLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(BiRNNLayer, self).__init__()
        self.bi_rnn = torch.nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, input_x, lens):
        output, _ = self.bi_rnn(input_x)
        avg_output = []
        for idx in range(len(lens)):
            avg = torch.mean(output[idx, :lens[idx], :], dim=0)
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
    def __init__(self, config, num_words,
                 pretrained_word_embedding=None, pretrained_label_embedding=None):
        super(HARNN, self).__init__()

        self.resnet = resnet34(pretrained=True, progress=True)
        self.embedding = EmbeddingLayer(word_num=num_words,
                                        embedding_dim=config["model"]["word_embedding_dim"],
                                        pretrained_word_matrix=pretrained_word_embedding)

        self.bi_rnn = BiRNNLayer(in_dim=config["model"]["word_embedding_dim"],
                                 hidden_dim=config["model"]["rnn_dim"],
                                 num_layers=1,
                                 dropout=config["model"]["dropout"])

        self.harl = HARL(rnn_dim=config["model"]["rnn_dim"],
                         image_feature_dim=self.resnet.fc.out_features,
                         label_embedding_dim=config["model"]["label_embedding_dim"],
                         num_classes_list=config["data"]["num_classes_list"],
                         pretrained_label_embedding=pretrained_label_embedding)

        # self.linear = torch.nn.Linear()
        # self.loss = Loss()

    def forward(self, image_input, text_record, segments, image_segments):
        segment_num_per_video = [sum(section) for section in segments]

        # 在同一个 section 之内的图像特征首尾帧相加并 / 2, 作为相应文本段的视频特征
        shape = image_input.shape
        # TODO: 视频分段保留原 RGB 图像, 不需要转化为灰度图像
        image_input = torch.randn(shape[0], 3, shape[1], shape[2]).to("cuda:3")
        image_feature = self.resnet(image_input)
        image_feature = (image_feature[:-1] + image_feature[1:]) / 2

        # 去掉跨 section 首尾相加的图像特征
        masked_indices = []
        offset = 0
        for video in image_segments:
            for idx in range(1, len(video) + 1):
                masked_indices.append(sum(video[:idx]) + offset - 1)
            offset += sum(video)
        # 去除最后一个 video 的最后一帧, 因为该帧并没有与来自其他 section 的帧相加
        masked_indices.pop(-1)
        mask = torch.BoolTensor([idx not in masked_indices for idx in range(image_feature.shape[0])])
        image_feature = image_feature[mask]

        text_pad, lens = text_record
        embedding_output = self.embedding(text_pad)
        rnn_out, rnn_avg = self.bi_rnn(embedding_output, lens)
        local_output_list, local_scores_list = self.harl(image_feature, rnn_out, rnn_avg, segments)
        local_scores = torch.cat(local_scores_list, dim=-1)
        local_output = torch.stack(local_output_list, dim=0)
        global_output = self.linear(local_output)
        
        pass


if __name__ == "__main__":
    pass
