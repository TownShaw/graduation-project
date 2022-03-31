'''
@Author: Xiao Tong
@FileName: harl.py
@CreateTime: 2022-03-22 20:34:21
@Description:

'''

import torch


class TCA(torch.nn.Module):
    def __init__(self, word_fearture_dim, label_feature_dim, num_classes):
        super(TCA, self).__init__()
        self.linear = torch.nn.Linear(word_fearture_dim, label_feature_dim, bias=False)
        self.S_h = torch.autograd.Variable(torch.randn(num_classes, label_feature_dim).type(torch.float32), requires_grad=True)

    def forword(self, word_x, last_w):
        V_h = torch.multiply(last_w, word_x)
        O_h = self.linear(V_h)
        attention_matrix = torch.matmul(self.S_h, O_h.transpose(1, 2))
        attention_weight = torch.softmax(attention_matrix, dim=-1)
        attention_out = torch.matmul(attention_weight, V_h)
        output = torch.mean(attention_out, dim=1)
        return output, attention_weight


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
    def __init__(self):
        super(CDM, self).__init__()

    def forward(self, P_L, atttention_weight):
        K_h = torch.mean(torch.multiply(atttention_weight, P_L))
        broadcast_dim = P_L.shape[-1]
        w_h = K_h.broadcast_to((broadcast_dim, K_h.shape[-1]))
        return w_h        


class HAM(torch.nn.Module):
    def __init__(self, rnn_hidden_dim, label_embedding_dim, num_classes):
        super(HAM, self).__init__()
        self.tca = TCA(word_fearture_dim=2 * rnn_hidden_dim, label_feature_dim=label_embedding_dim)
        self.cpm = CPM(in_channel=4 * rnn_hidden_dim, hidden_channel=rnn_hidden_dim, num_classes=num_classes)
        self.cdm = CDM()

    def forward(self, input_x, w_h_last, local_output_list, local_scores_list):
        r_att, W_att = self.tca(input_x, w_h_last)
        avg_input = torch.mean(input_x, dim=1)
        cpm_input = torch.cat(r_att, avg_input, dim=-1)
        A_L, P_L = self.cpm(cpm_input)
        w_h = self.cdm(P_L, W_att)
        local_output_list.append(A_L)
        local_scores_list.append(P_L)
        return input_x, w_h, local_output_list, local_scores_list


class HARL(torch.nn.Module):
    def __init__(self, rnn_dim, label_embedding_dim, num_classes_list, pretrained_label_embedding=None):
        super(HARL, self).__init__()
        self.net_sequence = []
        for num_classes in num_classes_list:
            self.net_sequence.append(HAM(rnn_dim, label_embedding_dim, num_classes))
        self.net_sequence = torch.nn.Sequential(*self.net_sequence)

    def forward(self, input_x):
        local_output_list = []
        local_scores_list = []
        # let w_0 = input_x
        self.net_sequence(input_x, input_x, local_output_list, local_scores_list)
        return local_output_list, local_scores_list


if __name__ == "__main__":
    pass
