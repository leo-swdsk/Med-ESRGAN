import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, channels, growth_channels):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet_CT(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=23, gc=32, scale=2): # in_nc = 1-> 1 Input Channel, out_nc = 1-> 1 Output Channel, nf = 64-> 64 Feature Maps after first convolution, nb = 23-> 23 Residual Dense Blocks, gc = 32-> 32 Growth Channels, scale = 2-> 2 Upsampling Factor
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1) # in_nc = 1-> 1 Input Channel, nf = 64-> 64 Feature Maps after first convolution, 3 = kernel size -> 3x3 , 1 = stride, 1 = padding
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling
        upsample_layers = []
        for _ in range(int(scale/2)):
            upsample_layers += [
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsampler = nn.Sequential(*upsample_layers)

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x) #input features (64 channels)
        trunk = self.trunk_conv(self.RRDB_trunk(fea)) #through the RRDB-Blocks
        fea = fea + trunk #Skip-Connection
        out = self.upsampler(fea) #Upsampling
        out = self.conv_last(out) #back to 1 channel (CT)
        return out
