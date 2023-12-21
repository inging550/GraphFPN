import torch.nn as nn
import torch.nn.functional as F
from net.backbone import Backbone
from net.graphlayer import contextual_layers, hierarchical_layers
import torch
from net.detect import Detect

# 图特征金字塔
class GraphFeaturePyramid(nn.Module):
  def __init__(self, bs, class_num, phi):
    """
    phi: YOLO8的尺度
    """
    super().__init__()
    self.bs = bs # batch_size
    self.class_num = class_num
    self.backbone, self.detect = backbone_detect(phi, class_num)
    self.stride = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])
    c3 = self.backbone.c3
    c4 = self.backbone.c4
    c5 = self.backbone.c5
    
    self.conv_c3_1x1 = nn.Conv2d(c3, 256, 1, 1)
    self.conv_c4_1x1 = nn.Conv2d(c4, 256, 1, 1)
    self.conv_c5_1x1 = nn.Conv2d(c5, 256, 1, 1)

    self.conv_c3_3x3 = nn.Conv2d(256, 256, 3, 1, 1)
    self.conv_c4_3x3 = nn.Conv2d(256, 256, 3, 1, 1)
    self.conv_c5_3x3 = nn.Conv2d(256, 256, 3, 1, 1)

    self.conv_c6_3x3 = nn.Conv2d(256, 256, 3 , 1, 1)
    self.conv_c7_3x3 = nn.Conv2d(256, 256, 3, 1, 1)
    self.upsample = nn.Upsample(scale_factor=2)

    self.context1 = contextual_layers(256, 256)
    self.context2 = contextual_layers(256, 256)
    self.context3 = contextual_layers(256, 256)
    self.hierarch1 = hierarchical_layers(256, 256)
    self.hierarch2 = hierarchical_layers(256, 256)
    self.hierarch3 = hierarchical_layers(256, 256)
    self.context4 = contextual_layers(256, 256)
    self.context5 = contextual_layers(256, 256)
    self.context6 = contextual_layers(256, 256)

  def forward(self, g, subg_c, subg_h, x):
    c3_out, c4_out, c5_out = self.backbone(x)
    p3_out = self.conv_c3_1x1(c3_out)
    p4_out = self.conv_c4_1x1(c4_out)
    p5_out = self.conv_c5_1x1(c5_out)
    # Graph operate
    p3_gnn = p3_out.permute(0, 2, 3, 1).reshape(-1, 256)        
    p4_gnn = p4_out.permute(0, 2, 3, 1).reshape(-1, 256)   
    p5_gnn = p5_out.permute(0, 2, 3, 1).reshape(-1, 256)
    p_final = torch.cat([p3_gnn, p4_gnn, p5_gnn], 0)
    g = cnn_gnn(g, p_final)
    nodes_update(subg_c, self.context1(subg_c, subg_c.ndata["pixel"]))
    nodes_update(subg_c, self.context2(subg_c, subg_c.ndata["pixel"]))
    nodes_update(subg_c, self.context3(subg_c, subg_c.ndata["pixel"]))
    nodes_update(subg_h, self.hierarch1(subg_h, subg_h.ndata["pixel"]))
    nodes_update(subg_h, self.hierarch2(subg_h, subg_h.ndata["pixel"]))
    nodes_update(subg_h, self.hierarch3(subg_h, subg_h.ndata["pixel"]))
    nodes_update(subg_c, self.context4(subg_c, subg_c.ndata["pixel"]))
    nodes_update(subg_c, self.context5(subg_c, subg_c.ndata["pixel"]))
    nodes_update(subg_c, self.context6(subg_c, subg_c.ndata["pixel"]))
    # data fusion
    p3_gnn, p4_gnn, p5_gnn = gnn_cnn(g, self.bs)
    p5_out = p5_out + p5_gnn.permute(0, 3, 1, 2)
    p4_out = p4_out + self.upsample(p5_out) + p4_gnn.permute(0, 3, 1, 2)
    p3_out = p3_out + self.upsample(p4_out) + p3_gnn.permute(0, 3, 1, 2)
    p5_out = p5_out 
    p3_out = self.conv_c3_3x3(p3_out)
    p4_out = self.conv_c4_3x3(p4_out)
    p5_out = self.conv_c5_3x3(p5_out)
    # p6_out = self.conv_c6_3x3(c5_out)
    # p7_out = self.conv_c7_3x3(F.relu(p6_out))
    return self.detect(p3_out, p4_out, p5_out)

def backbone_detect(phi, class_num):
  """
  从YOLO8中提取backbone以及detect, 并且导入预训练权重
  """
  depth_dict = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
  width_dict = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
  deep_width_dict = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
  backbone_len = {'n': 162, 's': 162, 'm': 234, 'l': 306, 'x': 306}
  path = "./model_data/yolov8{}.pt".format(phi)
  dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
  base_c = int(wid_mul * 64)
  base_d = max(round(dep_mul*3), 1)

  backbone = Backbone(base_c, base_d, deep_mul, path, backbone_len[phi])
  stride = torch.tensor([256 / x.shape[-2] for x in backbone.forward(torch.zeros(1, 3, 256, 256))])
  detect = Detect(class_num, base_c, deep_mul, stride)
  return backbone, detect

def cnn_gnn(g, c):
  g.ndata["pixel"] = c
  return g

def gnn_cnn(g, bs): 
  p3 = torch.reshape(g.ndata["pixel"][:6400*bs], (bs, 80, 80, 256))              # number of pixel in layers p3, 28*28 = 784
  p4 = torch.reshape(g.ndata["pixel"][6400*bs:8000*bs], (bs, 40, 40, 256))            # number of pixel in layers p4, 14*14 = 196
  p5 = torch.reshape(g.ndata["pixel"][8000*bs:8400*bs], (bs, 20, 20, 256))           # number of pixel in layers p5, 7*7 = 49
  return p3, p4, p5


def nodes_update(g, val):
  g.apply_nodes(lambda nodes: {'pixel' : val})
    


