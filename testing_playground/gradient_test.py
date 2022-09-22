from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        return x

def main():
    model = Net()
    sample_in = torch.rand(size=(4, 1))
    sample_out = torch.rand(size=(1,1))

    loss_fn = F.mse_loss()
    optimizer = optim.SGD(params=model.parameters(), lr = 0.003)

    out = model(sample_in)
    loss = loss_fn(out, sample_out)
    loss.backward()

    optimizer.step()