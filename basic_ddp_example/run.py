#!/usr/bin/env python
import os
from unittest import TestLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel.data_parallel as ddp
import torchvision
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms

# Copied this network from the official PyTorch MNIST example
# https://github.com/pytorch/examples/blob/main/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Use default transformation as used in https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
def get_mnist(is_train):
    mnist = torchvision.datasets.MNIST("../data", train=is_train, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    return mnist

def run(rank, size):
    """ Distributed function to be run on each GPU. """
    BATCHSIZE = 64
    torch.manual_seed(1234)
    model = Net().to(rank)
    ddp_model = ddp(model, device_ids = [rank], output_device = rank)

    # Now setup dataloader and standard training loop

    train_data = get_mnist(True)
    test_data = get_mnist(False)
    loss_fn = F.nll_loss()
    optimizer = torch.optim.SGD(ddp_model.model_parameters(), lr = 1e-3)  # Use standard SGD as baseline, can do Adam etc. later
    batchsz = BATCHSIZE // size

    train_sampler = DistributedSampler(train_data, num_replicas=size, rank=rank)
    test_sampler = DistributedSampler(test_data, num_replicas=size, rank=rank)  # Is distributed sampler here also necessary? -> not in bagua example, testing done only by one gpu???
    trainloader = DataLoader(train_data, shuffle=False, sampler=train_sampler, batch_size=batchsz )
    testloader = DataLoader(test_data, shuffle=False, sampler=test_sampler, batch_size=batchsz)





    train(trainloader)




def train(dataloader):
    # Implement training loop here
    pass






def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


# Use two GPUs in this example
if __name__ == "__main__":
    size = 2  # worldsize
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()