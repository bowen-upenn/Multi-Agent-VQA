import torch
import torch.distributed as dist
import os
import sys


def collate_fn(batch):
    """
    This function solves the problem when some data samples in a batch are None.
    :param batch: the current batch in dataloader
    :return: a new batch with all non-None data samples
    """
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))
