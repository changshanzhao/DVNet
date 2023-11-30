# 导入必要库
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        self.labels = []
        self.transform = transform

        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            file_list = os.listdir(class_dir)
            self.filepaths.extend([os.path.join(class_dir, file) for file in file_list])
            self.labels.extend([i] * len(file_list))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        file_path = self.filepaths[index]
        label = self.labels[index]
        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# 读取数据集
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # 在对图像进行标准化处理时，标准化参数来自于官网所提供的tansfer learning教程
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
root_dir = r'C:\Users\Lenovo\Desktop\process'
dataset = CustomDataset(root_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# 划分数据集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

# 加载数据集，设置batch_size
trainloader = DataLoader(train_set, batch_size=4, shuffle=True)
testloader = DataLoader(test_set, batch_size=1, shuffle=True)

# 定义损失函数为交叉熵损失函数
Lossfun = nn.CrossEntropyLoss()


# ResNet-18/34 残差结构 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1

    # 定义初始函数
    # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):  # downsample对应虚线残差结构捷径中的1×1卷积
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用bn层时不使用bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 实/虚线残差结构主分支中第二层stride都为1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # 默认是None

    # 定义正向传播过程
    def forward(self, x):
        identity = x  # 捷径分支的输出值
        if self.downsample is not None:  # 对应虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # 这里不经过relu激活函数

        out += identity
        out = self.relu(out)

        return out


# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_weights = self.conv(x)  # 通过卷积层得到注意力权重
        attention_weights = torch.sigmoid(attention_weights)  # 使用sigmoid函数将权重限制在0到1之间
        return x * attention_weights  # 将输入与注意力权重相乘，增强对应通道的重要性


# ResNet整个网络的框架部分
class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构，Basicblock or Bottleneck
                 blocks_num,  # 列表参数，所使用残差结构的数目，如对ResNet-34来说即是[3,4,6,3]
                 num_classes=1000,  # 训练集的分类个数
                 include_top=True):  # 为了能在ResNet网络基础上搭建更加复杂的网络，默认为True
        super(ResNet, self).__init__()
        self.include_top = include_top  # 传入类变量

        self.in_channel = 64  # 通过max pooling之后所得到的特征矩阵的深度

        self.conv1 = nn.Conv2d(4, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)  # 输入特征矩阵的深度为3（RGB图像），高和宽缩减为原来的一半
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 高和宽缩减为原来的一半

        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 对应conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)  # 对应conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)  # 对应conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)  # 对应conv5_x
        self.attention = AttentionModule(in_channels=512)  # 添加注意力模块

        if self.include_top:  # 默认为True
            # 无论输入特征矩阵的高和宽是多少，通过自适应平均池化下采样层，所得到的高和宽都是1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # num_classes为分类类别数

        for m in self.modules():  # 卷积层的初始化操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # stride默认为1
        # block即BasicBlock/Bottleneck
        # channel即残差结构中第一层卷积层所使用的卷积核的个数
        # block_num即该层一共包含了多少层残差结构
        downsample = None

        # 左：输出的高和宽相较于输入会缩小；右：输入channel数与输出channel数不相等
        # 两者都会使x和identity无法相加
        if stride != 1 or self.in_channel != channel * block.expansion:  # ResNet-18/34会直接跳过该if语句（对于layer1来说）
            # 对于ResNet-50/101/152：
            # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
            # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
            downsample = nn.Sequential(  # 下采样
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))  # 将特征矩阵的深度翻4倍，高和宽不变（对于layer1来说）

        layers = []
        layers.append(block(self.in_channel,  # 输入特征矩阵深度，64
                            channel,  # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):  # 从第二层开始都是实线残差结构
            layers.append(block(self.in_channel,  # 对于浅层一直是64，对于深层已经是64*4=256了
                                channel))  # 残差结构主分支上的第一层卷积的卷积核个数

        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)  # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回

    # 正向传播过程
    def forward(self, x):
        x = self.conv1(x)  # 7×7卷积层
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 3×3 max pool

        x = self.layer1(x)  # conv2_x所对应的一系列残差结构
        x = self.layer2(x)  # conv3_x所对应的一系列残差结构
        x = self.layer3(x)  # conv4_x所对应的一系列残差结构
        x = self.layer4(x)  # conv5_x所对应的一系列残差结构
        x = self.attention(x)  # 使用注意力模块

        if self.include_top:
            x = self.avgpool(x)  # 平均池化下采样
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


# 利用GPU进行训练
def resnet34(num_classes=6, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


net = resnet34()
# 定义优化器及一些超参数
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 迭代5轮
for epoch in range(2):
    # 初始化loss为0
    total_loss = 0.0
    for i, data in enumerate(trainloader):
        # 加载数据放到cuda里
        inputs, labels = data

        # 输入进模型，并计算loss
        pred = net(inputs)
        loss = Lossfun(pred, labels)
        # 梯度清空
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器优化
        optimizer.step()
        # 计算total_loss
        total_loss += loss.item()
        if i % 2 == 0:
            # 每轮打印输出一次
            print('[%d %d] loss:%.3f' % (epoch + 1, i + 1, total_loss/2))
        total_loss = 0.0

net.eval()
cnt = 0.0
# 用测试集测试并输出结果
for i, data in enumerate(testloader):
    inputs, labels = data

    # argmax取最高概率为结果
    outs = net(inputs).argmax()
    # 输出放回cpu比较一下，预测正确，cut+=1
    if outs == labels:
        cnt += 1
# 输出准确率
print(cnt / len(testloader))
torch.save(net.state_dict(), 'model.pth')
"""
[1] loss:1.049
[1] loss:0.257
[1] loss:3.419
[1] loss:0.228
[1] loss:0.218
[1] loss:1.721
[1] loss:0.811
[1] loss:1.481
[1] loss:1.120
[1] loss:0.482
[1] loss:0.538
[1] loss:0.228
[1] loss:0.177
[1] loss:3.096
[1] loss:0.070
[1] loss:0.127
[1] loss:0.097
[1] loss:2.381
[1] loss:1.482
[1] loss:0.584
[1] loss:0.537
[1] loss:0.403
[1] loss:2.176
[1] loss:2.145
[1] loss:1.761
[1] loss:0.345
[2] loss:0.642
[2] loss:0.773
[2] loss:0.471
[2] loss:1.254
[2] loss:0.383
[2] loss:0.254
[2] loss:1.716
[2] loss:0.187
[2] loss:0.273
[2] loss:1.771
[2] loss:1.706
[2] loss:0.842
[2] loss:0.806
[2] loss:1.204
[2] loss:0.281
[2] loss:1.414
[2] loss:0.168
[2] loss:2.023
[2] loss:1.929
[2] loss:0.271
[2] loss:1.368
[2] loss:0.501
[2] loss:0.890
[2] loss:0.620
[2] loss:0.544
[2] loss:1.390
0.2857142857142857
"""
'''
[1] loss:0.530
[1] loss:2.393
[1] loss:0.482
[1] loss:0.995
[1] loss:1.202
[1] loss:0.903
[1] loss:0.251
[1] loss:1.273
[1] loss:0.527
[1] loss:0.137
[1] loss:0.036
[1] loss:0.027
[1] loss:0.020
[1] loss:4.232
[1] loss:0.090
[1] loss:0.220
[1] loss:1.342
[1] loss:0.665
[1] loss:1.725
[1] loss:1.800
[1] loss:1.119
[1] loss:0.807
[1] loss:0.418
[1] loss:1.535
[1] loss:0.278
[1] loss:1.432
[1] loss:0.402
[2] loss:0.979
[2] loss:0.692
[2] loss:0.757
[2] loss:0.692
[2] loss:0.909
[2] loss:0.516
[2] loss:1.084
[2] loss:0.444
[2] loss:0.419
[2] loss:0.356
[2] loss:0.271
[2] loss:0.189
[2] loss:0.127
[2] loss:0.084
[2] loss:2.920
[2] loss:0.059
[2] loss:2.850
[2] loss:2.476
[2] loss:0.164
[2] loss:1.400
[2] loss:0.906
[2] loss:0.949
[2] loss:0.308
[2] loss:1.766
[2] loss:0.162
[2] loss:2.100
[2] loss:1.967
0.8571428571428571
'''
