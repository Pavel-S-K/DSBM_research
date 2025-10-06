import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cpu'


class NeuralODE(nn.Module):

    def __init__(self, dim, m):
        super(NeuralODE, self).__init__()
        m = 64
        self.net = nn.Sequential(
            nn.Linear(dim, m),
            nn.SiLU(),
            nn.Linear(m, m),
            nn.SiLU(),
            nn.Linear(m, dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)