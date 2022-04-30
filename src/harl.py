'''
@Author: Xiao Tong
@FileName: harl.py
@CreateTime: 2022-03-22 20:34:21
@Description:

'''

import torch
from torch.nn.init import xavier_normal_


class mySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class TCA(torch.nn.Module):
    def __init__(self, word_feature_dim, image_feature_dim, label_feature_dim, num_classes):
        super(TCA, self).__init__()
        self.linear_label = torch.nn.Linear(word_feature_dim, label_feature_dim, bias=False)
        self.linear_image = torch.nn.Linear(word_feature_dim, image_feature_dim, bias=False)
        self.S_h = torch.nn.parameter.Parameter(xavier_normal_(torch.randn(num_classes, label_feature_dim)).type(torch.float32), requires_grad=True)

    def forward(self, image_x, word_x, last_w):
        V_h = torch.multiply(last_w, word_x)
        class_O_h = torch.tanh(self.linear_label(V_h).transpose(1, 2))
        class_attention_matrix = torch.matmul(self.S_h, class_O_h)
        class_attention_weight = torch.softmax(class_attention_matrix, dim=-1)
        class_attention_out = torch.matmul(class_attention_weight, V_h)
        class_output = torch.mean(class_attention_out, dim=1)

        image_O_h = torch.tanh(self.linear_image(V_h).transpose(1, 2))
        image_attention_matrix = torch.matmul(image_x.unsqueeze(1), image_O_h)
        image_attention_weight = torch.softmax(image_attention_matrix, dim=-1)
        image_attention_out = torch.matmul(image_attention_weight, V_h)
        image_output = torch.mean(image_attention_out, dim=1)
        return class_output, image_output, class_attention_weight


class CPM(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, num_classes):
        super(CPM, self).__init__()
        self.linear1 = torch.nn.Linear(in_channel, hidden_channel)
        self.linear2 = torch.nn.Linear(hidden_channel, num_classes)

    def forward(self, input_x):
        A_L = torch.relu(self.linear1(input_x))
        P_L = torch.sigmoid(self.linear2(A_L))
        return A_L, P_L


class CDM(torch.nn.Module):
    def __init__(self, rnn_hidden_dim):
        super(CDM, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

    def forward(self, P_L, atttention_weight):
        K_h = torch.mean(torch.multiply(atttention_weight, P_L.unsqueeze(-1)), dim=1)
        w_h = K_h.unsqueeze(-1).repeat(1, 1, 2 * self.rnn_hidden_dim)
        return w_h


class HAM(torch.nn.Module):
    def __init__(self, rnn_hidden_dim, image_feature_dim, label_embedding_dim, fc_dim, num_classes):
        super(HAM, self).__init__()
        self.tca = TCA(word_feature_dim=2 * rnn_hidden_dim,
                       image_feature_dim=image_feature_dim,
                       label_feature_dim=label_embedding_dim,
                       num_classes=num_classes)
        self.cpm = CPM(in_channel=3 * 2 * rnn_hidden_dim + image_feature_dim, hidden_channel=fc_dim, num_classes=num_classes)
        self.cdm = CDM(rnn_hidden_dim=rnn_hidden_dim)

    def forward(self, image_feature, text_feature, text_avg, w_h_last, local_output_list, local_scores_list):
        r_att, image_att, W_att = self.tca(image_feature, text_feature, w_h_last)
        cpm_input = torch.cat([r_att, text_avg, image_att, image_feature], dim=-1)
        A_L, P_L = self.cpm(cpm_input)
        w_h = self.cdm(P_L, W_att)
        local_output_list.append(A_L)
        local_scores_list.append(P_L)
        return image_feature, text_feature, text_avg, w_h, local_output_list, local_scores_list


class HARL(torch.nn.Module):
    def __init__(self, rnn_dim, image_feature_dim, label_embedding_dim, fc_dim, num_classes_list, pretrained_label_embedding=None):
        super(HARL, self).__init__()
        self.net_sequence = []
        for num_classes in num_classes_list:
            self.net_sequence.append(HAM(rnn_dim, image_feature_dim, label_embedding_dim, fc_dim, num_classes))
        self.net_sequence = mySequential(*self.net_sequence)

    def forward(self, image_feature, rnn_out, rnn_avg):
        local_output_list = []
        local_scores_list = []
        # let w_0 = input_x
        self.net_sequence(image_feature, rnn_out, rnn_avg, rnn_out, local_output_list, local_scores_list)
        return local_output_list, local_scores_list


if __name__ == "__main__":
    pass
