from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.communication import recv
from bagua.torch_api.communication import send
from bagua.torch_api.communication import allreduce
import bagua.torch_api as bagua
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple

class AdasumAlgorithmImpl(AlgorithmImpl):
    def __init__(self, process_group: BaguaProcessGroup):
        super(AdasumAlgorithmImpl, self).__init__(process_group)

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_adasum(gradient: BaguaBucket, distance = 1):
            """
            train_data corresponds to a local gradient calculated on a microbatch. This functions combines
            the different gradients computed on each GPU individually to a global gradient.
            Microbatches are distrinct, i.e. share no data points, and together form one minibatch.
            """
            rank = bagua.get_rank() 
            world_size = bagua.get_world_size()

            # TODO: have to get length of BaguaBucket, i.e. number of elements. Assume its flattened for now
            
            

        return hook_adasum


class AdasumAlgorith (Algorithm):
    def __init__(self, hierarchical: bool = False):
        """
        Create an instance of the
        `QAdam Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .
        Args:
            q_adam_optimizer: A QAdamOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical

    def reify(self, process_group: BaguaProcessGroup) -> AdasumAlgorithmImpl:
        return AdasumAlgorithmImpl(process_group)