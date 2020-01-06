#!/usr/bin/env python3
# %%
import os
import torch.nn as nn
import torch.functional as F

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
        self.feat_dim = feat_dim
        self.num_target = num_target

        self.layer1 = nn.Conv1d(feat_dim, 512, 5, padding=2, dilation=1)
        self.bnorm1 = nn.BatchNorm1d(512)

        self.layer2 = nn.Conv1d(512, 512, 3, padding=2, dilation=2)
        self.bnorm2 = nn.BatchNorm1d(512)

        self.layer3 = nn.Conv1d(512, 512, 3, padding=3, dilation=3)
        self.bnorm3 = nn.BatchNorm1d(512)

        self.layer4 = nn.Conv1d(512, 512, 1, dilation=1)
        self.bnorm4 = nn.BatchNorm1d(512)

        self.layer5 = nn.Conv1d(512, 1500, 1, dilation=1)
        self.bnorm5 = nn.BatchNorm1d(512)

        self.pooling = StatsPooling(2)

        self.layer6 = nn.Linear(1500, 512)
        self.bnorm6 = nn.BatchNorm1d(512)

        self.layer7 = nn.Linear(512, 512)
        self.bnorm7 = nn.BatchNorm1d(512)

        self.layer8 = nn.Linear(512, num_target)

    def forward(self, data):
        x = self.bnorm1(self.layer1(data).relu)
        x = self.bnorm2(self.layer2(x).relu)
        x = self.bnorm3(self.layer3(x).relu)
        x = self.bnorm4(self.layer4(x).relu)
        x = self.bnorm5(self.layer5(x).relu)
        x = self.pooling(x)
        x = self.bnorm6(self.layer6(x).relu)
        x = self.bnorm7(self.layer7(x).relu)
        x = self.layer8(x)
        return x
