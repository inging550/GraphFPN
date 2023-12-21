# 定义检测头以及整体结构
import torch.nn as nn
import torch
from net.block import CBS
from net.backbone import Backbone
from net.neck import Neck
import math

  
class Detect(nn.Module):
  def __init__(self, class_num, base_c, deep_mul, stride):
    super(Detect, self).__init__()
    self.reg_max = 16
    self.stride = stride
    c_in = [256, base_c*8, int(base_c*16*deep_mul)]  # det123的通道数
    self.class_num = class_num
    self.out = [0] * 3
    c2, c3   = max((16, c_in[0] // 4, self.reg_max * 4)), max(c_in[0], class_num) 
    self.reg_conv = nn.ModuleList(nn.Sequential(
                                CBS(256, c2, 3, 1, 1),
                                CBS(c2, c2, 3, 1, 1),
                                nn.Conv2d(c2, 4*self.reg_max, 1, 1)
                                ) for i in range(3))
    self.cls_conv = nn.ModuleList(nn.Sequential(
                                  CBS(256, c3, 3, 1, 1),
                                  CBS(c3, c3, 3, 1, 1),
                                  nn.Conv2d(c3, class_num, 1, 1)
                                  ) for i in range(3))
    self.init_bias()  # 初始化bias
    
  def init_bias(self):
    for a, b, s in zip(self.reg_conv, self.cls_conv, self.stride):
      a[-1].bias.data[:] = 1.0
      b[-1].bias.data[:self.class_num] = math.log(5 / self.class_num / (640/s)**2)


  def forward(self, det1, det2, det3):
    """
    det1 [bs, 64, 80, 80]
    det2 [bs, 128, 40, 40]
    det3 [bs, 256, 20, 20]
    """
    # bs = det1.shape[0]
    # 得到三个det的输出
    for i,det in enumerate([det1, det2, det3]):
      self.out[i] = torch.cat((self.reg_conv[i](det), self.cls_conv[i](det)), 1)
    return self.out
  
