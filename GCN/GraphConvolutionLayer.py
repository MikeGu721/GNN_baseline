import torch
import torch.nn as nn
import torch.nn.init as init


class GraphConvolutionLayer(nn.Module):
    def __init__(self, node_size, feature_size, use_bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(node_size, node_size))
        if use_bias == True:
            self.bias = nn.Parameter(torch.Tensor(node_size, feature_size))
        self.use_bias = use_bias
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, x, adj):
        new_x = torch.mm(self.weight.T, x)
        new_x = torch.sparse.mm(adj, new_x)
        if self.use_bias == True:
            new_x += self.bias
        return new_x