import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, alpha, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.input_size = input_size
        self.output_size = output_size
        self.alpha = alpha
        self.concat = concat

        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.attention = nn.Parameter(torch.Tensor(2 * output_size, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def forward(self, x, adj):
        Wh = torch.mm(x, self.weight)
        # 相关度
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # 真正的attention weight
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def reset_parameters(self):
        init.xavier_uniform(self.weight.data, gain=1.414)
        init.xavier_uniform(self.attention.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.attention[:self.output_size, :])
        Wh2 = torch.matmul(Wh, self.attention[self.output_size:, :])

        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.input_size) + '->' + str(self.output_size) + ')'

