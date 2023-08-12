import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from linknet import BasicBlock,Encoder,Decoder

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class CALinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(CALinkNet, self).__init__()

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

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.coordAtt1 = CoordAtt(64, 64)
        self.coordAtt2 = CoordAtt(64, 64)
        self.coordAtt3 = CoordAtt(128, 128)
        self.coordAtt4 = CoordAtt(256, 256)

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
        c4 = self.coordAtt4(e3)
        l4 = c4 + d4
        d3 = self.decoder3(l4)
        c3 = self.coordAtt3(e2)
        l3 = c3 + d3
        d2 = self.decoder2(l3)
        c2 = self.coordAtt2(e1)
        l2 = c2 + d2
        d1 = self.decoder1(l2)
        c1 = self.coordAtt1(x)
        l1 = c1 + d1
        # d4 = e3 + self.decoder4(e4)
        # d3 = e2 + self.decoder3(d4)
        # d2 = e1 + self.decoder2(d3)
        # d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(l1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        y = self.lsm(y)

        return y