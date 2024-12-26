import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

import sys
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

class GCN_model2_bn(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_layers, dropout) :
        super().__init__()
        self.Conv1 = GCNConv(in_feat, h_feat)
        self.bn1 = torch.nn.BatchNorm1d(h_feat)
        self.Conv2 = GCNConv(h_feat, out_feat)
        self.clf1 = nn.Linear(out_feat, num_layers)
        self.dropout = dropout

    def forward(self, seq, edge_index):
        res1 = self.Conv1(seq, edge_index)
        res1 = self.bn1(res1)
        res1 = F.relu(res1)
        res1 = F.dropout(res1, p=self.dropout, training = self.training)
        res2 = self.Conv2(res1, edge_index)
        res3 = self.clf1(res2)

        return res3

