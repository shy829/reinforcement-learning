import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np


# noise not used temporarily
# use Dueling network, use advantage net and value net to calculate Q
from utils import preprocess


class Net(nn.Module):
    def __init__(self, n_actions, LR, EPS):
        super(Net, self).__init__()
        self.n_actions = n_actions
        # network layer with 3 convolutional layers
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # advantage layer with 2 fully_connected layers
        self.adv = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )
        # value layer with 2 fully_connected layers
        self.val = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=LR, eps=EPS)
        self.mls = nn.MSELoss()

    def forward(self, s):
        s = self.fc(s)
        # (batch_size，channels，x，y)->(batch_size, channels*x*y)
        s = s.view(s.size(0), -1)
        adv = self.adv(s)
        val = self.val(s)
        s = val + adv - adv.mean()
        return s

    def normalize(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.0)
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def save_checkpoint(self):
        print("... saving checkpoint")
        path = os.getcwd() + '/result'
        checkpoint_file = os.path.join(path, 'DDQN')
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint")
        path = os.getcwd() + '/result'
        checkpoint_file = os.path.join(path, 'DDQN')
        self.load_state_dict(torch.load(checkpoint_file))
