import numpy as np
import random
import torch
from mmcv.utils import Registry

# Create a register of dataset、loss and activation
ACTIVATION = Registry('activation')
LOSS = Registry('loss')
DATASET = Registry('dataset')


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)