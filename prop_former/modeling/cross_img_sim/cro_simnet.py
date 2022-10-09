import torch
import torch.nn as nn
import numpy as np
import queue
from prop_former.modeling.fc_modules import ResidualFullyConnectedBranch


class BalanceBinaryWeightManager(object):
    def __init__(self):
        self.neg_num_queue = queue.deque(maxlen=25)
        self.pos_num_queue = queue.deque(maxlen=25)
        self.neg_num_queue.append(1)
        self.pos_num_queue.append(1)

        return

    def update(self, GT_map):
        self.neg_num_queue.append((GT_map[:, ::5, ::5, ] == 0).sum().item())
        self.pos_num_queue.append((GT_map[:, ::5, ::5, ] == 1).sum().item())
        return

    def get_balance_weight(self):
        neg_num = sum(self.neg_num_queue)
        pos_num = sum(self.pos_num_queue)

        neg_w = pos_num / (pos_num + neg_num)
        pos_w = neg_num / (pos_num + neg_num)

        return neg_w, pos_w


class CroPixelSimConvNet(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int,
                 layer_num=3, func='sigmoid', batch_norm=True):
        super(CroPixelSimConvNet, self).__init__()
        self.func = func

        self.layers = nn.Sequential()

        dim_in = in_feature

        for l in range(layer_num):
            self.layers.add_module(f'Conv{l}', nn.Conv2d(dim_in, hidden_size, kernel_size=1))
            if batch_norm:
                self.layers.add_module(f'BN{l}', nn.BatchNorm2d(hidden_size))

            self.layers.add_module(f'RL{l}', nn.ReLU(inplace=True))
            dim_in = hidden_size

        if self.func == 'sigmoid':
            self.layers.add_module(f'Out{l}', nn.Conv2d(dim_in, 1, kernel_size=1))
            self.layers.add_module(f'Sigmoid{l}', nn.Sigmoid())
        elif self.func == 'softmax':
            self.layers.add_module(f'Out{l}', nn.Conv2d(dim_in, 2, kernel_size=1))
        else:
            raise NotImplementedError

    def forward(self, x):

        if self.func == 'sigmoid':
            res = self.layers(x)
        elif self.func == 'softmax':
            feat = self.layers(x)
            res = torch.softmax(feat, dim=1)[:, 1][:, None]
        else:
            raise NotImplementedError

        return res


class CroPixelResSimConvNet(nn.Module):
    def __init__(self, in_dim, feat_dim, layer_num=3, use_bn=True):
        super(CroPixelResSimConvNet, self).__init__()
        self.fc_branch = ResidualFullyConnectedBranch(in_dim, [feat_dim for l in range(layer_num)], use_bn=use_bn)
        self.out_head = nn.Conv2d(feat_dim, 2, kernel_size=1)

    def forward(self, x):
        feat = self.fc_branch(x)
        logit = self.out_head(feat)
        res = torch.softmax(logit, dim=1)[:, 1][:, None]
        return res


def get_cro_simnet(cfg, dim_in, dim_mid):
    layer_num = cfg.CROSS_IMG_SIM.LayerNum
    batch_norm = cfg.CROSS_IMG_SIM.BN
    net = CroPixelResSimConvNet(dim_in, dim_mid, layer_num, use_bn=batch_norm)
    return net
