import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

class contextual_layers(nn.Module):
  def __init__(self, in_dim, h_dim):
    super().__init__()
    self.in_dim = in_dim
    self.gat1 = GATConv(in_dim, h_dim, 1, activation=F.relu)
    self.gat2 = GATConv(in_dim, h_dim, 1, activation=F.relu)
    self.gat3 = GATConv(in_dim, h_dim, 1, activation=F.relu)

  def forward(self, g, f):
    f = self.gat1(g, f)
    f = self.gat2(g, f)
    f = self.gat3(g, f)
    return f.squeeze()
  
class hierarchical_layers(nn.Module):
  def __init__(self, in_dim, h_dim):
    super().__init__()
    self.in_dim = in_dim
    self.gat1 = GATConv(in_dim, h_dim, 1, activation=F.relu)
    self.gat2 = GATConv(in_dim, h_dim, 1, activation=F.relu)
    self.gat3 = GATConv(in_dim, h_dim, 1, activation=F.relu)

  def forward(self, g, f):
    f = self.gat1(g, f)
    f = self.gat2(g, f)
    f = self.gat3(g, f)
    return f.squeeze()


    