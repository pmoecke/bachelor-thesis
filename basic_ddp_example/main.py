#!/usr/bin/env python
# Use different approach in this file, the other one produced an error regarding data_parallel(), which 
# evaluates a model in parallel
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(model, train_loader, optimizer, epoch):
    log_interval = 10
    dry_run = False

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def run(rank, world_size):
    """ Distributed function to be run on each GPU. """
    # Manually set batch size for now, later parsed
    BATCHSIZE = 64
    torch.manual_seed(1234)

    # Now setup dataloader and standard training loop

    # This ensures that the data is only downloaded once
    if rank == 0:
        train_data = get_mnist(True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        train_data = get_mnist(True)
    
    test_data = get_mnist(False)
    batchsz = BATCHSIZE // world_size  # Split up batch to all gpus

    train_kwargs = {"batch_size": batchsz}
    test_kwargs = {"batch_size": BATCHSIZE}
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}  # Need to test out different settings
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_kwargs.update(
        {
            "sampler": train_sampler,
            "shuffle": False,
        }
    )
    trainloader = DataLoader(train_data, **train_kwargs)  # loog up how exactly the **kwargs args are passed, i.e meaning of **
    testloader = DataLoader(test_data, **test_kwargs)

    model = Net().to(rank)
    ddp_model = ddp(model, device_ids = [rank])
    optimizer = torch.optim.SGD(ddp_model.model_parameters(), lr = 1e-3)  # Use standard SGD as baseline, can do Adam etc. later

    epochs = 20
    for epoch in range(1, epochs + 1):
        train(ddp_model, trainloader, optimizer, epoch)
        test(ddp_model, testloader)

    cleanup()

def main():
    # Set the number of gpus manually for now, later the argument should be able to be parsed when executing the script
    world_size = 2
    mp.spawn(run, args=(world_size), nprocs=world_size, join=True)  # read documentation to understand how processes are spawned


# Use two GPUs in this example
if __name__ == "__main__":
    main()