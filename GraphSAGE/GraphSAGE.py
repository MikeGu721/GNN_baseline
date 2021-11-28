import torch.nn as nn
from layer import SageGCN


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim=None,
                 num_neighbors_list=None):
        '''
        初始化GraphSAGE模型
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param num_neighbors_list: 聚合节点数
        '''
        super(GraphSage, self).__init__()
        if num_neighbors_list is None:
            num_neighbors_list = [10, 10]
        if hidden_dim is None:
            hidden_dim = [64, 64]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)

        self.gcn = []
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        self.gcn.append(SageGCN(hidden_dim[0], hidden_dim[1], activation=None))

    def forward(self, node_features_list):
        '''

        :param node_features_list: 是一个列表，表示每个node的feature
        :return:
        '''
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - 1):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view(src_node_num, self.num_neighbors_list[hop], -1)
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]  # 第0个元素表示源节点的特征
