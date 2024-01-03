# 思路
## backbone
backbone部分使用yolov8的backbone, 输出80*80, 40*40, 20*20三个feature map =>p3 p4 p5

## neck
neck部分待用GraphFPN, 对三个feature map进行处理
1、用1x1卷积更改通道数为256,然后将特征放入预先设定好的图结构中
2、使用GAT提取特征，contexture *3 + hierarchial*3 + contexeure*3
3、将图结构转变为fature map格式 =>p3gnn p4gnn p5gnn

对三个输出进行后处理
p5 = p5 + p5gnn
p4 = p4 + upsample(p5) + p4gnn
p3 = p3 + upsample(p4) + p3gnn
再通过3x3卷积到输出

## head
head部分采用yolov8的head对p3 p4 p5进行处理的得到8400个bbox，双向选择后计算loss

# 训练结果
在训练初期10epoch左右，设定conf=0.73,iou=0.7 , mAP50=0.174  mAP50-95=0.113
许多negative样本预测框conf高于0.7，当conf>0.73后许多就会消除许多negative样本
其对小目标以及残缺目标的识别精度尚可，对大样本识别不佳
会对一些奇怪的特征点进行过渡识别，一般为像素梯度较大的地方
在ground truth周围会存在许多正确的预测框 
对像素值平滑的特征识别到位，比如天空，就不会过渡识别

epoch=20 conf=0.73, iou=0.7 mAP50=0.196 mAP50-95=0.127

epoch=30 conf=0.73, iou=0.7 mAP50=0.212 mAP50-95=0.136 
若将iou减小到0.1 肉眼识别效果会好很多,但是map性能下降

epoch=40 mAP50=0.216 mAP50-95=0.137

epoch=80 mAP50=0.209 mAP50-95=0.128
0.243 0.140 conf=0.7 iou=0.5
