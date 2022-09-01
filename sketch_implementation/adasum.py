from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.communication import recv
from bagua.torch_api.communication import send
from bagua.torch_api.communication import allreduce
from . import env
from .env import (
    get_master_addr,
    get_world_size,
    get_rank,
    get_local_rank,
    get_node_rank,
    get_default_bucket_size,
    get_bagua_service_port,
    get_autotune_server_wait_time,
    find_free_network_port,
)
from torch.optim.optimizer import Optimizer
import torch
import math
from typing import List, Tuple

class AdasumAlgorithmImpl(AlgorithmImpl):
    def __init__(self, process_group: BaguaProcessGroup):
        super(AdasumAlgorithmImpl, self).__init__(process_group)

    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook_adasum(gradient: BaguaBucket, distance):
            """
            train_data corresponds to a local gradient calculated on a microbatch. This functions combines
            the different gradients computed on each GPU individually to a global gradient.
            Microbatches are distrinct, i.e. share no data points, and together form one minibatch.
            """
            rank = get_rank() 
            world_size = get_world_size()

            # TODO: have to get length of BaguaBucket, i.e. number of elements. Assume its flattened??


        return hook_adasum


class AdasumAlgorith (Algorithm):
    def __init__(self, q_adam_optimizer: QAdamOptimizer, hierarchical: bool = True):
        """
        Create an instance of the
        `QAdam Algorithm <https://tutorials.baguasys.com/algorithms/q-adam>`_
        .
        Args:
            q_adam_optimizer: A QAdamOptimizer initialized with model parameters.
            hierarchical: Enable hierarchical communication.
        """
        self.hierarchical = hierarchical
        self.optimizer = q_adam_optimizer

    def reify(self, process_group: BaguaProcessGroup) -> AdasumAlgorithmImpl:
        return AdasumAlgorithmImpl(process_group)