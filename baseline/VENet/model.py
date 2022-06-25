'''
@Author: Tong Xiao
@FileName: model.py
@CreateTime: 2022-06-24 12:58:06
@Description:

'''

import torch
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


class Conv2dWithMaxPool(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: int,
                 maxpool_kernel_size: int,
                 conv_stride: int=1,
                 maxpool_stride: int=1,
                 conv_padding: int=0,
                 maxpool_padding: int=0) -> None:
        super(Conv2dWithMaxPool, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding)
        self.maxpool = torch.nn.MaxPool2d(maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    def forward(self, input_x):
        return self.maxpool(self.conv2d(input_x))


class F2CAtt(torch.nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super(F2CAtt, self).__init__()
        self.W_f2c = torch.nn.Linear(feature_dim, feature_dim // 2, bias=False)
        self.V_f2c = torch.nn.Linear(feature_dim // 2, 1, bias=False)
        self.tanh = torch.nn.Tanh()

    def forward(self, pr_f, r_c):
        pr_f_copy = pr_f.view(pr_f.shape[0], pr_f.shape[1], -1).transpose(1, 2)
        feature = torch.cat([pr_f_copy, r_c.unsqueeze(1).repeat(1, pr_f_copy.shape[1], 1)], dim=-1)
        att_matrix = self.V_f2c(self.tanh(self.W_f2c(feature)))
        alpha = torch.softmax(att_matrix.squeeze(-1), dim=-1)
        alpha = alpha.view(pr_f.shape[0], 1, pr_f.shape[2], pr_f.shape[3])
        return torch.mul(alpha, pr_f)


class SRN(torch.nn.Module):
    def __init__(self,
                 word_num: int,
                 word_embeding_dim: int,
                 text_lstm_dim: int,
                 fig_lstm_dim: int,
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_sizes: list,
                 maxpool_kernel_sizes: list,
                 pretrained_word_matrix=None) -> None:
        super(SRN, self).__init__()
        # 2 layer CNN
        assert len(conv_kernel_sizes) == 2
        assert len(conv_kernel_sizes) == len(maxpool_kernel_sizes)

        self.embedding = EmbeddingLayer(
            word_num=word_num,
            embedding_dim=word_embeding_dim,
            pretrained_word_matrix=pretrained_word_matrix
        )

        self.fig_cnn = torch.nn.Sequential(*[
            Conv2dWithMaxPool(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_kernel_size=conv_kernel_sizes[0],
                maxpool_kernel_size=maxpool_kernel_sizes[0],
                maxpool_stride=maxpool_kernel_sizes[0]
            ),
            Conv2dWithMaxPool(
                in_channels=out_channels,
                out_channels=out_channels,
                conv_kernel_size=conv_kernel_sizes[1],
                maxpool_kernel_size=maxpool_kernel_sizes[1],
                maxpool_stride=maxpool_kernel_sizes[1]
            )
        ])

        self.text_lstm = torch.nn.LSTM(
            input_size=word_embeding_dim,
            hidden_size=text_lstm_dim,
            batch_first=True,
            bidirectional=False
        )

        self.f2catt = F2CAtt(out_channels + text_lstm_dim)

        self.hlstm = torch.nn.LSTM(
            input_size=out_channels,
            hidden_size=fig_lstm_dim,
            batch_first=True,
            bidirectional=False
        )
        self.vlstm = torch.nn.LSTM(
            input_size=fig_lstm_dim,
            hidden_size=fig_lstm_dim,
            batch_first=True,
            bidirectional=False
        )

        default_kernel = 3
        self.att_cnn = Conv2dWithMaxPool(
            in_channels=out_channels,
            out_channels=out_channels,
            conv_kernel_size=default_kernel,
            maxpool_kernel_size=default_kernel,
            maxpool_stride=default_kernel
        )

    def forward(self, images, text_record):
        # text
        text, lens = text_record
        embed = self.embedding(text)
        text_pack = pack_padded_sequence(embed, lens, batch_first=True, enforce_sorted=False)
        text_lstm_out, _ = self.text_lstm(text_pack)
        text_lstm_out, lens = pad_packed_sequence(text_lstm_out, batch_first=True)
        r_c = text_lstm_out[:, -1, :]

        # image
        pr_f = self.fig_cnn(images)

        r_att_f = self.f2catt(pr_f, r_c)
        lstm_in = r_att_f.transpose(1, 2).transpose(2, 3)
        hlstm_out, _ = self.hlstm(lstm_in.reshape(-1, lstm_in.shape[2], lstm_in.shape[3]))
        r_row_t = hlstm_out[:, -1, :].view(lstm_in.shape[0], lstm_in.shape[1], -1)
        vlstm_out, _ = self.vlstm(r_row_t)
        r_f_t = vlstm_out[:, -1, :]
        r_s_t = torch.mean(self.att_cnn(r_att_f), dim=[2, 3])
        return torch.cat([r_s_t, r_f_t, r_c], dim=-1)


class FusionCNN(torch.nn.Module):
    def __init__(self, feature_dim: int, out_channels: int, kernel_size_list: list) -> None:
        super(FusionCNN, self).__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(
                1,
                out_channels,
                kernel_size=(w, feature_dim),
                padding=(w // 2, 0)
            )
        for w in kernel_size_list])

    def forward(self, input_x, segments):
        convs_out = []
        start_idx, end_idx = 0, 0
        for segment in segments:
            start_idx = end_idx
            end_idx += sum(segment)
            tmp_input = input_x[start_idx:end_idx]
            out = [conv(tmp_input.view(1, 1, *tmp_input.shape)) for conv in self.convs]
            out = torch.cat(out, dim=0).squeeze(-1)
            out = out.reshape(out.shape[-1], out.shape[0], out.shape[1])
            convs_out.append(out)
        convs_out = torch.cat(convs_out, dim=0)
        return convs_out


class VENet(torch.nn.Module):
    def __init__(self, config: dict, word_num: int, pretrained_word_embedding=None) -> None:
        super(VENet, self).__init__()
        self.srn = SRN(
            word_num=word_num,
            word_embeding_dim=config["model"]["word_embedding_dim"],
            text_lstm_dim=config["model"]["text_lstm_dim"],
            fig_lstm_dim=config["model"]["fig_lstm_dim"],
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            conv_kernel_sizes=config["model"]["kernel_sizes"],
            maxpool_kernel_sizes=config["model"]["maxpool_sizes"],
            pretrained_word_matrix=pretrained_word_embedding
        )

        d_1 = config["model"]["out_channels"] + config["model"]["fig_lstm_dim"] + config["model"]["text_lstm_dim"]

        self.fusion_cnn = FusionCNN(
            feature_dim=d_1,
            out_channels=config["model"]["d_2"],
            kernel_size_list=config["model"]["fusion_kernels"]
        )

        self.mlp = torch.nn.Sequential(*[
            torch.nn.Linear(len(config["model"]["fusion_kernels"] * config["model"]["d_2"]), config["model"]["fc_dim"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["model"]["fc_dim"], config["data"]["num_classes"])
        ])

    def forward(self, images, text_record, segments):
        srn_out = self.srn(images, text_record)
        fusion_out = self.fusion_cnn(srn_out, segments)
        logits = self.mlp(fusion_out.view(fusion_out.shape[0], -1))
        segment_scores = torch.sigmoid(logits)
        scores = []
        start_idx, end_idx = 0, 0
        for segment in segments:
            start_idx = end_idx
            end_idx += sum(segment)
            scores.append(torch.max(segment_scores[start_idx:end_idx], dim=0)[0])
        scores = torch.stack(scores, dim=0)
        return scores
