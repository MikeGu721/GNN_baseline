from GraphConvolutionLayer import GraphConvolutionLayer
import torch.nn as nn
import torch
import scipy.sparse as sp
import numpy as np


class GraphConvolutionNetwork(nn.Module):
    def __init__(self, layer_num, node_size, feature_size):
        super(GraphConvolutionNetwork, self).__init__()
        self.GCLs = []
        for i in range(layer_num):
            self.GCLs.append([GraphConvolutionLayer(node_size, feature_size), nn.ReLU()])

    def forward(self, x, adj):
        for GCL, relu in self.GCLs:
            x = GCL(x, adj)
            x = relu(x)
        return x

def normalization(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree,-0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

if __name__ == '__main__':
    node_size = 20
    feature_size = 70

    GCN = GraphConvolutionNetwork(1, node_size, feature_size)
    x = torch.rand(node_size, feature_size)
    adj = torch.randint(2, (node_size, node_size))
    adj = torch.FloatTensor(adj.numpy())
    x = GCN(x, adj)
    print(x)
