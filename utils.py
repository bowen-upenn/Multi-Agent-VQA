import torch
import torch.distributed as dist
import os
import sys


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def collate_fn(batch):
    """
    This function solves the problem when some data samples in a batch are None.
    :param batch: the current batch in dataloader
    :return: a new batch with all non-None data samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))
