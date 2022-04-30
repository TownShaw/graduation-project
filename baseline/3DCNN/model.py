'''
@Author: Xiao Tong
@FileName: model.py
@CreateTime: 2022-04-30 20:01:50
@Description:

'''

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


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
        # iuput: torch.Size([batch_size, 3, 16, 224, 224])
        # output: torch.Size([batch_size, 100, 14, 218, 218])
        self.cnn_3d = torch.nn.Conv3d(3, out_channel, (3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, image_input):
        convs_out = self.cnn_3d(image_input.transpose(1, 2))
        pool_x = F.max_pool3d(convs_out, (3, 8, 8), stride=(1, 8, 8), padding=(1, 0, 0))
        return self.dropout(pool_x)


class CNN3DLSTM(torch.nn.Module):
    def __init__(self, config, num_words, pretrained_word_embedding=None):
        super(CNN3DLSTM, self).__init__()
        self.embedding = EmbeddingLayer(num_words,
                                        config["model"]["word_embedding_dim"],
                                        pretrained_word_embedding)

        self.bi_rnn = BiRNNLayer(config["model"]["word_embedding_dim"],
                                 config["model"]["rnn_dim"],
                                 config["model"]["rnn_num_layers"],
                                 config["model"]["dropout"])

        self.cnn3d = CNN3D(out_channel=config["model"]["out_channel"],
                           kernel_size_list=config["model"]["kernel_size"],
                           dropout=config["model"]["dropout"])

        self.linear_out = torch.nn.Linear(2 * config["model"]["rnn_dim"] + (14 * 14) * config["model"]["out_channel"],
                                          config["data"]["num_classes"])

    def forward(self, image_input, text_record, segments, image_segments):
        start_idx, end_idx = 0, 0
        batched_images = []
        pixel_num = image_input.shape[-1] * image_input.shape[-2]
        for video in image_segments:
            start_idx = end_idx
            end_idx += sum(video)
            batched_images.append(image_input[start_idx:end_idx])
        batched_images = pad_sequence(batched_images, batch_first=True)
        batch_max_len = batched_images.shape[1]
        image_lens = [sum(video) for video in image_segments]

        image_output = self.cnn3d(batched_images)

        image_avg = []
        for idx, video in enumerate(image_lens):
            image_avg.append(image_output[idx, :, :video, :, :])
        image_avg = torch.cat(image_avg, dim=1).transpose(0, 1)
        image_avg = (image_avg[:-1] + image_avg[1:]) / 2

        # 去掉跨 section 首尾相加的图像特征
        masked_indices = []
        offset = 0
        for video in image_segments:
            for idx in range(1, len(video) + 1):
                masked_indices.append(sum(video[:idx]) + offset - 1)
            offset += sum(video)
        # 去除最后一个 video 的最后一帧, 因为该帧并没有与来自其他 section 的帧相加
        masked_indices.pop(-1)
        mask = torch.BoolTensor([idx not in masked_indices for idx in range(image_avg.shape[0])])
        image_avg = image_avg[mask]
        image_avg = image_avg.reshape(image_avg.shape[0], -1)

        input_x, lens = text_record
        embedding_out = self.embedding(input_x)
        rnn_avg = self.bi_rnn(embedding_out, lens)
        logits = self.linear_out(torch.cat([image_avg, rnn_avg], dim=-1))
        scores = torch.sigmoid(logits)

        start_idx, end_idx = 0, 0
        video_scores = []
        for video in segments:
            start_idx = end_idx
            end_idx += sum(video)
            video_scores.append(torch.max(scores[start_idx:end_idx], dim=0)[0])
        video_scores = torch.stack(video_scores, dim=0)
        return video_scores
