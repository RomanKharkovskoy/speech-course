import logging

import torch
import torch.nn as nn
from thop import profile

from melbanks import LogMelFilterBanks

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, n_mels=80, groups=1):
        super().__init__()

        self.melbanks = LogMelFilterBanks(n_mels=n_mels)

        self.conv1 = nn.Conv1d(n_mels, 64, kernel_size=3, padding=1, groups=groups)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=min(groups, 64))
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.melbanks(x)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x.squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape=(1, 1, 16000)):
    dummy = torch.randn(input_shape)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops
