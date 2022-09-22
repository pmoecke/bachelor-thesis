from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup, ReduceOp
from bagua.torch_api.communication import new_group
import bagua.torch_api as bagua
import torch
import math
from typing import List, Tuple


def get_flattened_tensor(tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 0:
            return

        total_size = 0
        for tensor in tensors:
            total_size += tensor.numel()

        flatten_tensor = torch.zeros(
            total_size, dtype=tensors[0].dtype, device=tensors[0].device
        )

        offset = 0
        for tensor in tensors:
            # copy data
            flatten_tensor[offset : offset + tensor.numel()] = tensor.reshape(-1)
            offset += tensor.numel()

        return flatten_tensor

class AllreduceAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self, 
        process_group: BaguaProcessGroup,
        sgd_optimizer: torch.optim.SGD
    ):
        super(AllreduceAlgorithmImpl, self).__init__(process_group)
        self.optimizer = sgd_optimizer
        self.hierarchical = False

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=True,
            group=self.process_group,
        )

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        
        def hook():
            world_size = bagua.get_world_size()
            comm = bagua.communication._get_default_group().get_global_communicator()
            print("hello")
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    # print(param.grad)
                    # as_list = get_flattened_tensor([param.grad])
                    # print(as_list.size())
                    # buffer = torch.zeros(size=as_list.size())
                    # print(buffer.size())

                    # bagua.allreduce(send_tensor=as_list, recv_tensor=buffer, ReduceOp=ReduceOp.AVG, comm=self.process_group.get_global_communicator)  # TODO: throws an error or something
                    # print(buffer)

                    send_tensor = torch.rand(288, dtype=torch.float32).cuda()
                    # print(grp = self.process_group.group_name)
                    # print(ranks = self.process_group.ranks)
                    # bagua.allreduce_inplace(tensor=send_tensor, ReduceOp=ReduceOp.AVG, comm=comm)

        return hook


class AllreduceAlgorithm (Algorithm):
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

    def reify(self, process_group: BaguaProcessGroup) -> AllreduceAlgorithmImpl:
        return AllreduceAlgorithmImpl(
            process_group,
            sgd_optimizer=self.optimizer
        )