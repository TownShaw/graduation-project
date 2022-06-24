'''
@Author: Xiao Tong
@FileName: harnn.py
@CreateTime: 2022-03-22 20:36:18
@Description:

'''

import torch
from harl import HARL
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.nn.init import xavier_normal_
from torchvision.models import resnet34
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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
        return output, avg_output


class Transformer(torch.nn.Module):
    """
    tranformer encoder
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int=2,
                 dropout: float=0.1,
                 layer_norm_eps: float=0.00001) -> None:
        super(Transformer, self).__init__()
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                                 nhead=nhead,
                                                                 dropout=dropout,
                                                                 layer_norm_eps=layer_norm_eps)
        self.encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_encoder_layers, self.encoder_norm)

    def forward(self, input_x, lens):
        output = self.transformer_encoder(input_x)
        avg_output = []
        lens = lens.tolist()
        for idx in range(len(lens)):
            avg = torch.mean(output[idx, :lens[idx], :], dim=0)
            avg_output.append(avg)
        avg_output = torch.stack(avg_output, dim=0)
        return output, avg_output


class HierarchyLossWithSegments(torch.nn.Module):
    def __init__(self):
        super(HierarchyLossWithSegments, self).__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, section_scores, video_scores, labels, segments):
        video_scores_from_sections = []
        start_idx, end_idx = 0, 0
        for video in segments:
            start_idx = end_idx
            end_idx += len(video)
            video_scores_from_sections.append(torch.max(section_scores[start_idx:end_idx], dim=0)[0])
        video_scores_from_sections = torch.stack(video_scores_from_sections, dim=0)

        video_loss = self.bce_loss(video_scores, labels)
        video_loss_from_sections = self.bce_loss(video_scores_from_sections, labels)
        total_loss = video_loss + video_loss_from_sections
        return total_loss


class HARNN(torch.nn.Module):
    def __init__(self, config, num_words,
                 pretrained_word_embedding=None, pretrained_label_embedding=None):
        super(HARNN, self).__init__()

        self.resnet = resnet34(pretrained=True, progress=True)
        self.embedding = EmbeddingLayer(word_num=num_words,
                                        embedding_dim=config["model"]["word_embedding_dim"],
                                        pretrained_word_matrix=pretrained_word_embedding)

        if config["model"]["Encoder"] == "Bi-RNN":
            self.encoder = BiRNNLayer(in_dim=config["model"]["word_embedding_dim"],
                                      hidden_dim=config["model"]["rnn_dim"],
                                      num_layers=config["model"]["rnn_num_layers"],
                                      dropout=config["model"]["dropout"])
            self.harl = HARL(word_feature_dim=2 * config["model"]["rnn_dim"],
                            image_feature_dim=self.resnet.fc.out_features,
                            label_embedding_dim=config["model"]["label_embedding_dim"],
                            num_classes_list=config["data"]["num_classes_list"],
                            fc_dim=config["model"]["fc_dim"],
                            pretrained_label_embedding=pretrained_label_embedding)
        elif config["model"]["Encoder"] == "Transformer":
            self.encoder = Transformer(d_model=config["model"]["word_embedding_dim"],
                                       nhead=config["model"]["nhead"],
                                       num_encoder_layers=config["model"]["num_encoder_layers"])
            self.harl = HARL(word_feature_dim=config["model"]["word_embedding_dim"],
                            image_feature_dim=self.resnet.fc.out_features,
                            label_embedding_dim=config["model"]["label_embedding_dim"],
                            num_classes_list=config["data"]["num_classes_list"],
                            fc_dim=config["model"]["fc_dim"],
                            pretrained_label_embedding=pretrained_label_embedding)
        else:
            raise ValueError("Encoder must be 'Bi-RNN' or 'Transformer'!")


        self.section_linear1 = torch.nn.Linear(config["model"]["fc_dim"], config["model"]["fc_dim"])
        self.section_linear2 = torch.nn.Linear(config["model"]["fc_dim"], config["data"]["num_classes"])
        self.video_linear1 = torch.nn.Linear(config["model"]["fc_dim"], config["model"]["fc_dim"])
        self.video_linear2 = torch.nn.Linear(config["model"]["fc_dim"], config["data"]["num_classes"])
        self.alpha = config["model"]["alpha"]

    def forward(self, image_input, text_record, segments, image_segments):
        # 在同一个 section 之内的图像特征首尾帧相加并 / 2, 作为相应文本段的视频特征
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
        text_feature_out, text_feature_avg = self.encoder(embedding_output, lens)
        segments_local_output_list, segments_local_scores_list = self.harl(image_feature, text_feature_out, text_feature_avg)

        segments_local_scores = torch.cat(segments_local_scores_list, dim=-1)
        segments_local_output = torch.stack(segments_local_output_list, dim=1)

        # 计算 section-level 的 scores
        section_local_output, section_local_scores = [], []
        start_idx, end_idx = 0, 0
        for video in segments:
            for idx in range(len(video)):
                start_idx = end_idx
                end_idx += video[idx]
                section_local_output.append(torch.mean(segments_local_output[start_idx:end_idx], dim=0))
                section_local_scores.append(torch.max(segments_local_scores[start_idx:end_idx], dim=0)[0])
        section_local_output = torch.stack(section_local_output, dim=0)
        section_local_scores = torch.stack(section_local_scores, dim=0)

        # 计算 video-level 的 scores
        video_local_output, video_local_scores = [], []
        start_idx, end_idx = 0, 0
        for video in segments:
            start_idx = end_idx
            end_idx += len(video)
            video_local_output.append(torch.mean(section_local_output[start_idx:end_idx], dim=0))
            video_local_scores.append(torch.max(section_local_scores[start_idx:end_idx], dim=0)[0])
        video_local_output = torch.stack(video_local_output, dim=0)
        video_local_scores = torch.stack(video_local_scores, dim=0)

        section_global_output = torch.relu(self.section_linear1(torch.mean(section_local_output, dim=1)))
        section_global_scores = torch.sigmoid(self.section_linear2(section_global_output))

        video_global_output = torch.relu(self.section_linear1(torch.mean(video_local_output, dim=1)))
        video_global_scores = torch.sigmoid(self.section_linear2(video_global_output))

        # section_final_scores 预测的是每个 section 的知识点, video_final_scores 预测的是每个 video 的知识点
        section_final_scores = (1 - self.alpha) * section_local_scores + self.alpha * section_global_scores
        video_final_scores = (1 - self.alpha) * video_local_scores + self.alpha * video_global_scores

        return section_final_scores, video_final_scores


if __name__ == "__main__":
    pass
