import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel.data_parallel as ddp
import torchvision
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms

# Current issues: how to extract the current gradients of the local model? param.grad.data
# Also, the tensor seems to have some specific structure, how structured? 
# Need to pass one-dimensional gradient (duh) to adasum 

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

def adasum_rvh(gradient:torch.Tensor, distance):
    """
    train_data corresponds to a local gradient calculated on a microbatch. This functions combines
    the different gradients computed on each GPU individually to a global gradient.
    Microbatches are distrinct, i.e. share no data points, and together form one minibatch.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # May need to create some buffers, how ensure that they are free'd after using?
    # assert world_size > 2 and power of 2
    mid = math.floor(len(gradient) / 2)
    # TODO: How to know how large the buffers needs to be??? Assume for now that they are the same size
    right_half = gradient[mid:].clone().cuda()
    left_half = gradient[:mid].clone().cuda()

    if math.floor(rank / distance) % 2 == 0:
        neighbor = rank + distance
        dist.send(tensor=right_half, dst=neighbor)         # Send right half              
        dist.recv(tensor=left_half, src=neigbor)           # Receive left half
    else:
        neigbor = rank - distance
        dist.send(tensor=left_half, dst=neighbor)
        dist.recv(tensor=right_half, src=neighbor)

    new_distance = 2 * distance
    partial_dots = torch.tensor([torch.dot(right_half, right_half), torch.dot(right_half, left_half), 
                                    torch.dot(left_half, left_half)]).cuda()

    grp = dist.new_group(ranks=[math.floor(rank / new_distance) * new_distance + i for i in range(0, new_distance - 1)])  # When to kill this group??
    partial_dots = dist.all_reduce(group=grp, op=dist.ReduceOp.SUM)
    # Now apply adasum
    new_gradient = left_half * (1 - (partial_dots[0] / 2 * partial_dots[1])) + right_half * (1 - (partial_dots[2] / 2 * partial_dots[1])).cuda()
    if new_distance < world_size:
        new_gradient = adasum_rvh(new_gradient, new_distance)
    
    dist.send(tensor=new_gradient, dst=neighbor)  # TODO
    other_half = gradient.clone().cuda()
    dist.recv(tensor=other_half, src=neighbor)
    if math.floor(rank / distance) % 2 == 0:
        gradient = torch.cat(new_gradient, other_half)
    else:
        gradient = torch.cat(other_half, new_gradient)
    
    return gradient  # Is it alright for it to be in-place???

def cleanup():
    dist.destroy_process_group()

def train(model, train_loader, optimizer, epoch, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()  # Compute local gradient

        # How to apply Adasum?
        for param in model.parameters():
            param.grad.data = adasum_rvh(param.grad.data, 1)  # This would spawn much to many sub-processes, communication channels

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test():
    pass

def run(rank, world_size, args):
    setup(rank, world_size)
    torch.manual_seed(args.seed)

    if rank == 0:
        train_data = get_mnist(True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        train_data = get_mnist(True)
    
    test_data = get_mnist(False)
    batchsz = args.batch_size // world_size  # Split up batch to all gpus
    train_kwargs = {"batch_size": batchsz}
    test_kwargs = {"batch_size": args.batch_size}
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

    model = Net().cuda(rank)







    cleanup()

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpus', type=int, default=1, metavar='N',
                        help='Number of GPUs')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # Assume in following that cuda is always available and we have one node with multiple gpus

    world_size = args.gpus
    print(f"Using {world_size} GPUs from {torch.cuda.device_count()} available GPUs")

    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()




