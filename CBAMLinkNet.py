import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from linknet import BasicBlock,Encoder,Decoder

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Res外部、第一层第二层加入CBAM
class CBAMLinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(CBAMLinkNet, self).__init__()

        base = resnet.resnet18(pretrained=True)

        # self.in_block = nn.Sequential(
        #     base.conv1,
        #     base.bn1,
        #     base.relu,
        #     base.maxpool
        # )

        self.conv1 = nn.Conv2d(25, 64, 7, 2, 3, bias=False)
        self.in_block = nn.Sequential(
            # base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.in_block(x)

        x = self.ca(x) * x
        x = self.sa(x) * x
        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.ca1(e4) * e4
        e4 = self.sa1(e4) * e4

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y

# 预训练 上采样加入四个
class CBAMLinkNet1(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(CBAMLinkNet1, self).__init__()

        base = resnet.resnet18(pretrained=True)

        # self.in_block = nn.Sequential(
        #     base.conv1,
        #     base.bn1,
        #     base.relu,
        #     base.maxpool
        # )

        self.conv1 = nn.Conv2d(25, 64, 7, 2, 3, bias=False)
        self.in_block = nn.Sequential(
            # base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # 网络的第一层加入注意力机制
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(64)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()
        self.ca4 = ChannelAttention(256)
        self.sa4 = SpatialAttention()

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = self.decoder4(e4)
        ca4 = self.ca4(d4) * d4
        sa4 = self.sa4(ca4) * ca4
        l4 = e3 + sa4
        d3 = self.decoder3(l4)
        ca3 = self.ca3(d3) * d3
        sa3 = self.sa3(ca3) * ca3
        l3 = e2 + sa3
        d2 = self.decoder2(l3)
        ca2 = self.ca2(d2) * d2
        sa2 = self.sa2(ca2) * ca2
        l2 = e1 + sa2
        d1 = self.decoder1(l2)
        ca1 = self.ca2(d1) * d1
        sa1 = self.sa2(ca1) * ca1
        l1 = x + sa1

        # Classifier
        y = self.tp_conv1(l1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        # y = self.lsm(y)

        return y

# 无预训练
class CBAMlinknetbase1(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(CBAMlinknetbase1, self).__init__()

        base = resnet.resnet18(pretrained=True)

        # self.in_block = nn.Sequential(
        #     base.conv1,
        #     base.bn1,
        #     base.relu,
        #     base.maxpool
        # )

        self.conv1 = nn.Conv2d(25, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder(64, 64, 3, 1, 1)
        self.encoder2 = Encoder(64, 128, 3, 2, 1)
        self.encoder3 = Encoder(128, 256, 3, 2, 1)
        self.encoder4 = Encoder(256, 512, 3, 2, 1)

        # 网络的第一层加入注意力机制
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(64)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()
        self.ca4 = ChannelAttention(256)
        self.sa4 = SpatialAttention()

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        d4 = self.decoder4(e4)
        ca4 = self.ca4(d4) * d4
        sa4 = self.sa4(ca4) * ca4
        l4 = e3 + sa4
        d3 = self.decoder3(l4)
        ca3 = self.ca3(d3) * d3
        sa3 = self.sa3(ca3) * ca3
        l3 = e2 + sa3
        d2 = self.decoder2(l3)
        ca2 = self.ca2(d2) * d2
        sa2 = self.sa2(ca2) * ca2
        l2 = e1 + sa2
        d1 = self.decoder1(l2)
        ca1 = self.ca2(d1) * d1
        sa1 = self.sa2(ca1) * ca1
        l1 = x + sa1

        # Classifier
        y = self.tp_conv1(l1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y
