import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from aggregation import NeighborAggregator


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, aggr_neighbor_method='mean', aggr_hidden_method='sum'):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ['mean', 'sum', 'max']
        assert aggr_hidden_method in ['sum', 'concat']
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        '''
        聚合+权重相乘
        :param src_node_features:
        :param neighbor_node_features:
        :return:
        '''
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden == 'sum':
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == 'concat':
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden
