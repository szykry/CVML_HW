import torch
import torch.nn as nn


class Conv(nn.Module):

    def __init__(self, in_channels, channels, k_size=3, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, channels, k_size, stride=stride, padding=k_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))


class ConvNet(nn.Module):

    def __init__(self, base_channels=16, in_channels=3,
                 num_classes=55):  # base: csatorna növekedéshez, in: RGB, 52 jó osztály
        super().__init__()

        # Filters
        self.c11 = Conv(in_channels, base_channels)
        self.c12 = Conv(base_channels, base_channels)
        self.d1 = Conv(base_channels, base_channels * 2, stride=2)
        # Downscale using strided convolution and expand channels

        self.c21 = Conv(base_channels * 2, base_channels * 2)
        self.c22 = Conv(base_channels * 2, base_channels * 2)
        self.d2 = Conv(base_channels * 2, base_channels * 4, stride=2)

        self.c31 = Conv(base_channels * 4, base_channels * 4)
        self.c32 = Conv(base_channels * 4, base_channels * 4)
        self.d3 = Conv(base_channels * 4, base_channels * 8, stride=2)

        self.c41 = Conv(base_channels * 8, base_channels * 8)
        self.c42 = Conv(base_channels * 8, base_channels * 8)
        self.d4 = Conv(base_channels * 8, base_channels * 16, stride=2)

        self.c51 = Conv(base_channels * 16, base_channels * 16)
        self.c52 = Conv(base_channels * 16, base_channels * 16)
        self.d5 = Conv(base_channels * 16, base_channels * 32, stride=2)

        # Input image is 32x32 -> after 5 downscaling the activation map is 1x1
        # [batch, ch, h, w]  -> [batch, ch] linearisba, h=1, w=1
        # Classifier is a normal 1x1 convolution that produces num_classes class scores
        # This layer does not have BatchNorm or ReLU

        self.classifier = nn.Conv2d(base_channels * 32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.d1(self.c12(self.c11(x)))
        x = self.d2(self.c22(self.c21(x)))
        x = self.d3(self.c32(self.c31(x)))
        x = self.d4(self.c42(self.c41(x)))
        x = self.d5(self.c52(self.c51(x)))

        return torch.squeeze(self.classifier(x))  # After squeeze is becomes (batch_size x num_classes)
