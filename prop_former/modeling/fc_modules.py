import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, d_in, d_out, use_bn):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Conv2d(d_in, d_out, kernel_size=1, )
        self.layer2 = nn.Conv2d(d_out, d_out, kernel_size=1, )
        self.use_bn = use_bn

        if use_bn:
            self.bn1 = nn.BatchNorm2d(d_out)
            self.bn2 = nn.BatchNorm2d(d_out)

        if d_in != d_out:
            self.sqz = nn.Conv2d(d_in, d_out, kernel_size=1, )
        else:
            self.sqz = None

    def forward(self, x):
        if self.sqz:
            residual = F.relu(self.sqz(x))
        else:
            residual = x

        x = self.layer1(x)
        if self.use_bn:
            x = self.bn1(x)

        x = F.relu(x)

        x = self.layer2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)

        x += residual
        return x


class ResidualFullyConnectedBranch(nn.Module):
    def __init__(self, feat_dim_in, dim_layer_list, use_bn):
        super(ResidualFullyConnectedBranch, self).__init__(),
        self.layers = nn.Sequential()

        d_in = dim_layer = feat_dim_in
        for i, dim_layer in enumerate(dim_layer_list):
            self.layers.add_module(f'block{i}', BasicBlock(d_in, dim_layer, use_bn))
            d_in = dim_layer

        self.feat_dim_out = dim_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
