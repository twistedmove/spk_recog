#!/usr/bin/env python3
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):
    def __init__(self, in_dim, out_dim, kennel_size, padding, dilation):
        super(TDNN, self).__init__()
        self.layer = nn.Conv1d(in_dim, out_dim, kennel_size,
                               padding=padding, dilation=dilation)
        self.bnorm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bnorm(F.relu(self.layer(x)))


class StatsPooling(nn.Module):
    def __init__(self, axis):
        super(StatsPooling, self).__init__()
        self.axis = axis

    def forward(self, data):
        mean = data.mean(dim=self.axis)
        std = data.pow(2).mean(self.axis).sqrt()
        x = torch.cat((mean, std), 1)
        return x


class Xvector(nn.Module):
    def __init__(self, feat_dim, num_target):
        super(Xvector, self).__init__()
        self.TDNN1 = nn.Conv1d(feat_dim, 512, 5, padding=2, dilation=1)
        self.TDNN2 = nn.Conv1d(512, 512, 3, padding=2, dilation=2)
        self.TDNN3 = nn.Conv1d(512, 512, 3, padding=3, dilation=3)
        self.TDNN4 = nn.Conv1d(512, 512, 1, dilation=1)
        self.TDNN5 = nn.Conv1d(512, 1500, 1, dilation=1)

        self.pooling = StatsPooling(2)

        self.layer1 = nn.Linear(1500, 512)
        self.bnorm1 = nn.BatchNorm1d(512)

        self.layer2 = nn.Linear(512, 512)
        self.bnorm2 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, num_target)

    def forward(self, data):
        x = self.TDNN1(data)
        x = self.TDNN2(x)
        x = self.TDNN3(x)
        x = self.TDNN4(x)
        x = self.TDNN5(x)
        x = self.pooling(x)
        x = self.bnorm1(F.relu(self.layer1(x)))
        x = self.bnorm2(F.relu(self.layer2(x)))
        x = self.output(x)
        return x
