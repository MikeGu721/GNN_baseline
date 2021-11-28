import torch.nn as nn
from GraphAttentionNetwork import GraphAttentionLayer
import torch
import torch.nn.functional as F


class GraphAttentionNetwork(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GraphAttentionNetwork, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, alpha, dropout, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, alpha, dropout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    nfeat = 20
    nhid = 100
    nclass = 5
    dropout = 0.1
    alpha = 0.9
    nheads = 2

    nsize = 34
    gat = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)
    x = torch.rand(nsize, nfeat)
    print(x.shape)
    adj = torch.randint(2, (nsize, nsize))
    adj = torch.FloatTensor(adj.numpy())
    x = gat(x, adj)
    print(x.shape)
