'''
@Author: Xiao Tong
@FileName: harnn.py
@CreateTime: 2022-03-22 20:36:18
@Description:

'''

import torch
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.nn.init import xavier_normal_
from torchvision.models import resnet34
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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
        return output, avg_output


class Transformer(torch.nn.Module):
    """
    tranformer encoder
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int=6,
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


class TCA(torch.nn.Module):
    def __init__(self, word_feature_dim, label_feature_dim, num_classes):
        super(TCA, self).__init__()
        self.linear_label = torch.nn.Linear(word_feature_dim, label_feature_dim, bias=False)
        self.S_h = torch.nn.parameter.Parameter(xavier_normal_(torch.randn(num_classes, label_feature_dim)).type(torch.float32), requires_grad=True)

    def forward(self, word_x):
        class_O_h = torch.tanh(self.linear_label(word_x).transpose(1, 2))
        class_attention_matrix = torch.matmul(self.S_h, class_O_h)
        class_attention_weight = torch.softmax(class_attention_matrix, dim=-1)
        class_attention_out = torch.matmul(class_attention_weight, word_x)
        class_output = torch.mean(class_attention_out, dim=1)
        return class_output


class ForwardUnit(torch.nn.Module):
    def __init__(self, image_feature_dim, word_feature_dim, label_feature_dim, hidden_dim, feature_out_dim, num_classes):
        super(ForwardUnit, self).__init__()
        self.tca = TCA(
            word_feature_dim=word_feature_dim,
            label_feature_dim=label_feature_dim,
            num_classes=num_classes
        )
        self.linear1 = torch.nn.Linear(image_feature_dim + word_feature_dim + hidden_dim, feature_out_dim)
        self.linear2 = torch.nn.Linear(feature_out_dim, feature_out_dim)
        self.linear3 = torch.nn.Linear(feature_out_dim, num_classes)

    def forward(self, image_x, word_x, last_h, local_scores_list: list):
        tca_out = self.tca(word_x)
        unit_input = torch.cat([image_x, tca_out, last_h], dim=-1)
        hidden_out = torch.relu(self.linear1(unit_input))
        local_feature = torch.relu(self.linear2(hidden_out))
        local_scores = torch.sigmoid(self.linear3(local_feature))
        local_scores_list.append(local_scores)
        return image_x, word_x, hidden_out, local_scores_list


class HMCN_F(torch.nn.Module):
    def __init__(self,
                 image_feature_dim: int,
                 word_feature_dim: int,
                 label_feature_dim: int,
                 hidden_dim: int,
                 num_classes_list: list):
        super(HMCN_F, self).__init__()
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.label_feature_dim = label_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes_list = num_classes_list
        self.network = []
        for idx in range(len(self.num_classes_list)):
            if idx == 0:
                self.network.append(ForwardUnit(
                    image_feature_dim=self.image_feature_dim,
                    word_feature_dim=self.word_feature_dim,
                    label_feature_dim=self.label_feature_dim,
                    hidden_dim=0,
                    feature_out_dim=self.hidden_dim,
                    num_classes=self.num_classes_list[idx]
                ))
            else:
                self.network.append(ForwardUnit(
                    image_feature_dim=self.image_feature_dim,
                    word_feature_dim=self.word_feature_dim,
                    label_feature_dim=self.label_feature_dim,
                    hidden_dim=self.hidden_dim,
                    feature_out_dim=self.hidden_dim,
                    num_classes=self.num_classes_list[idx]
                ))
        self.network = mySequential(*self.network)

    def forward(self, image_x, word_x):
        local_scores_list = []
        _, __, global_feature, local_scores_list = self.network(
            image_x,
            word_x,
            torch.FloatTensor().to(word_x.device),
            local_scores_list
        )
        return global_feature, local_scores_list


class HMCNModel(torch.nn.Module):
    def __init__(self, config: dict, num_words: int, pretrained_word_embedding=None) -> None:
        super(HMCNModel, self).__init__()
        self.alpha = config["model"]["alpha"]
        self.resnet = resnet34(pretrained=True, progress=True)
        self.embedding = EmbeddingLayer(word_num=num_words,
                                        embedding_dim=config["model"]["word_embedding_dim"],
                                        pretrained_word_matrix=pretrained_word_embedding)
        if config["model"]["Encoder"] == "Bi-RNN":
            self.encoder = BiRNNLayer(in_dim=config["model"]["word_embedding_dim"],
                                    hidden_dim=config["model"]["rnn_dim"],
                                    num_layers=config["model"]["rnn_num_layers"],
                                    dropout=config["model"]["dropout"])
            self.hmcn = HMCN_F(
                image_feature_dim=self.resnet.fc.out_features,
                word_feature_dim=2 * config["model"]["rnn_dim"],
                label_feature_dim=config["model"]["label_embedding_dim"],
                hidden_dim=config["model"]["hidden_dim"],
                num_classes_list=config["data"]["num_classes_list"]
            )
        elif config["model"]["Encoder"] == "Transformer":
            self.encoder = Transformer(d_model=config["model"]["word_embedding_dim"],
                                       nhead=config["model"]["nhead"],
                                       num_encoder_layers=config["model"]["num_encoder_layers"])
            self.hmcn = HMCN_F(
                image_feature_dim=self.resnet.fc.out_features,
                word_feature_dim=config["model"]["word_embedding_dim"],
                label_feature_dim=config["model"]["label_embedding_dim"],
                hidden_dim=config["model"]["hidden_dim"],
                num_classes_list=config["data"]["num_classes_list"]
            )
        self.section_linear = torch.nn.Linear(config["model"]["hidden_dim"], config["data"]["num_classes"])
        self.video_linear = torch.nn.Linear(config["model"]["hidden_dim"], config["data"]["num_classes"])

    def forward(self, images, text_record, segments, image_segments):
        # 在同一个 section 之内的图像特征首尾帧相加并 / 2, 作为相应文本段的视频特征
        image_feature = self.resnet(images)
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
        text_feature_out, avg_out = self.encoder(embedding_output, lens)
        segment_global_feature, segments_local_scores_list = self.hmcn(image_feature, text_feature_out)

        segments_local_scores = torch.cat(segments_local_scores_list, dim=-1)

        # 计算 section-level 的 scores
        section_global_feature, section_local_scores = [], []
        start_idx, end_idx = 0, 0
        for video in segments:
            for idx in range(len(video)):
                start_idx = end_idx
                end_idx += video[idx]
                section_global_feature.append(torch.mean(segment_global_feature[start_idx:end_idx], dim=0))
                section_local_scores.append(torch.max(segments_local_scores[start_idx:end_idx], dim=0)[0])
        section_global_feature = torch.stack(section_global_feature, dim=0)
        section_local_scores = torch.stack(section_local_scores, dim=0)

        # 计算 video-level 的 scores
        video_global_feature, video_local_scores = [], []
        start_idx, end_idx = 0, 0
        for video in segments:
            start_idx = end_idx
            end_idx += len(video)
            video_global_feature.append(torch.mean(section_global_feature[start_idx:end_idx], dim=0))
            video_local_scores.append(torch.max(section_local_scores[start_idx:end_idx], dim=0)[0])
        video_global_feature = torch.stack(video_global_feature, dim=0)
        video_local_scores = torch.stack(video_local_scores, dim=0)

        section_global_scores = torch.sigmoid(self.section_linear(section_global_feature))

        video_global_scores = torch.sigmoid(self.video_linear(video_global_feature))

        # section_final_scores 预测的是每个 section 的知识点, video_final_scores 预测的是每个 video 的知识点
        section_final_scores = (1 - self.alpha) * section_local_scores + self.alpha * section_global_scores
        video_final_scores = (1 - self.alpha) * video_local_scores + self.alpha * video_global_scores

        return section_final_scores, video_final_scores


if __name__ == "__main__":
    pass
