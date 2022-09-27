from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from bagua.torch_api.communication import new_group
import bagua.torch_api as bagua
import torch
import math
from typing import List, Tuple

def main():
    test = torch.rand(size=(100, 100)).cuda()
    buf =  torch.zeros(size=(100, 100)).cuda()
    print(test)

    bagua.init_process_group()

    if bagua.get_rank() == 0:
        bagua.send(tensor=test, dst=1)
        bagua.recv(tensor=buf, src=1)
    else:
        bagua.send(tensor=test, dst=0)
        bagua.recv(tensor=buf, src=0)

    print(buf)

if __name__ == "__main__":
    main()

