from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.communication import recv
from bagua.torch_api.communication import send
from bagua.torch_api.communication import allreduce
from bagua.torch_api.communication import new_group
import bagua.torch_api as bagua
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple

class AdasumAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self, 
        process_group: BaguaProcessGroup,
        sgd_optimizer: torch.optim.SGD
    ):
        super(AdasumAlgorithmImpl, self).__init__(process_group)
        self.optimizer = sgd_optimizer

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_adasum():
            """
            Still need to write
            """
            def adasum(torch.Tensor: gradient, distance, world_size):
                distance = 1
                rank = bagua.get_rank()
                mid = math.floor(gradient.size()[0]/2)  # split current gradient across first dimension

                # Save the left and right dimensions and use flag such that we know which dimension the
                # buffer of the last receive operation needs to have
                dim_left = gradient[0:mid].size()
                dim_right = gradient[mid:,].size()
                is_left = True

                # Can use half of input gradient as buffer for the other half which a process receives (?)
                grad_a = torch.Tensor(0)
                grad_b = torch.Tensor(0)

                if math.floor(rank/distance) % 2 == 0:              # process right half
                    neighbor = rank + distance
                    grad_b = torch.zeros(size=dim_right).cuda()
                    send(tensor=gradient[mid:,], dst=neighbor)
                    recv(tensor=grad_b, src=neighbor)               # override the half of input which we do not use, could also use a separate buffer
                    grad_a = gradient[0:mid]
                else:                                               # process left half
                    neighbor = rank - distance
                    grad_a = torch.zeros(size=dim_left).cuda()
                    send(tensor=gradient[0:mid], dst=neighbor)
                    recv(tensor=grad_a, src=neighbor)
                    grad_b = gradient[mid:,]
                    is_left = False

                new_distance = 2 * distance
                dim_cnt = len(own_half.size())
                dot_product = torch.tensordot(grad_a, grad_b, dims=dim_cnt)
                norm_a = torch.tensordot(grad_a, grad_a, dims=dim_cnt)
                norm_b = torch.tensordot(grad_b, grad_b, dims=dim_cnt)
                partial_dots = torch.Tensor([dot_product, norm_a, norm_b]).cuda()

                group = new_group(ranks=[math.floor(rank / new_distance) * new_distance + i for i in range(0, new_distance - 1)])
                all_dots = torch.zeros(size=partial_dots.size()).cuda()
                allreduce(send_tensor=partial_dots, recv_tensor=all_dots, comm=group.get_global_communicator())

                new_gradient = grad_a * (1 - (all_dots[0] / 2 * all_dots[1])) + grad_b * (1 - (all_dots[0] / 2 * all_dots[2])).cuda()

                if new_distance < world_size:
                    new_gradient = adasum(new_gradient, new_distance, world_size)
                
                send(tensor=new_gradient, dst=neighbor)
                recv_buf = torch.Tensor(0)
                if is_left:
                    recv_buf = torch.zeros(size=dim_right)
                else:
                    recv_buf = torch.zeros(size=dim_left)
                recv(tensor=recv_buf, src=neighbor)

                if math.floor(rank / distance) % 2 == 0:
                    return torch.cat(tensors=(new_gradient, recv_buf), dim=0)
                else:
                    return torch.cat(tensors=(recv_buf, new_gradient), dim=0)

            world_size = bagua.get_world_size()
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    # Maybe need a synchronization point here
                    param.grad = adasum(param.grad, world_size)  # Do I need to overwrite the gradients??      

        return hook_adasum


class AdasumAlgorith (Algorithm):
    def __init__(self, sgd_optimizer: torch.optim.SGD, hierarchical: bool = False):
        """
        Create an instance of the
        `Adasum Algorithm <https://arxiv.org/abs/2006.02924>`
        .
        Args:
            sgd_optimizer: A regular SGD optimizer from PyTorch initialized with model parameters.
            hierarchical: Enable hierarchical communication. (not implemented for now)
        """
        self.optimizer = sgd_optimizer
        self.hierarchical = hierarchical

    def reify(self, process_group: BaguaProcessGroup) -> AdasumAlgorithmImpl:
        return AdasumAlgorithmImpl(
            process_group,
            sgd_optimizer=self.optimizer
        )