# 定义Backbone结果
import torch.nn as nn
from net.block import C2F, CBS, SPFF
import torch

class Backbone(nn.Module):
  def __init__(self, base_c, base_d, deep_mul, path, len_backbone):
    super(Backbone, self).__init__()
    # 这里按照执行顺序定义Backbone
    self.conv1 = CBS(3, base_c, 3, 2, 1)
    self.conv2 = CBS(base_c, base_c*2, 3, 2, 1)
    self.c2f1 = C2F(base_c*2, base_c*2, base_d, True)
    self.conv3 = CBS(base_c*2, base_c*4, 3, 2, 1)
    self.c2f2 = C2F(base_c*4, base_c*4, base_d*2, True) # out->feature1
    self.conv4 = CBS(base_c*4, base_c*8, 3, 2, 1)
    self.c2f3 = C2F(base_c*8, base_c*8, base_d*2, True) # out->feature2
    self.conv5 = CBS(base_c*8, int(base_c*16*deep_mul), 3, 2, 1)
    self.c2f4 = C2F(int(base_c*16*deep_mul), int(base_c*16*deep_mul), base_d, True)
    self.spff = SPFF(int(base_c*16*deep_mul)) # out->feature3
    self.c3 = base_c * 4
    self.c4 = base_c * 8
    self.c5 = int(base_c*16*deep_mul)
    self.load_pretrain(path, len_backbone)

  def load_pretrain(self, path, len_backbone):
    pretrain_weight = list(torch.load(path)['model'].state_dict().items())
    self_weight = self.state_dict()
    for index, key in enumerate(self_weight.keys()):
      block_name = key.split('.')[-1]
      if pretrain_weight[index][0].endswith(block_name) and index<len_backbone:
        self_weight[key] = pretrain_weight[index][1]
    self.load_state_dict(self_weight)

  def forward(self, x):
    """
    input [3,640,640]
    output:
      feature1:
      feature2:
      feature3:
    """
    x = self.conv2(self.conv1(x))
    x = self.conv3(self.c2f1(x))
    feature1 = self.c2f2(x) # [bs, 64, 80, 80]
    x = self.conv4(feature1)
    feature2 = self.c2f3(x) # [bs, 128, 40, 40]
    x = self.c2f4(self.conv5(feature2)) 
    feature3 = self.spff(x) # [bs, 256, 20, 20]
    return feature1, feature2, feature3