'''
邻居聚合功能
'''
import torch.nn as nn
import torch.nn.init as init
import torch


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method='mean'):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        assert neighbor_feature in ['mean', 'sum', 'max']
        if self.aggr_method == 'mean':
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == 'sum':
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == 'max':
            aggr_neighbor = neighbor_feature.max(dim=1)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden
