import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import yaml
import os
import json
import argparse

from utils import *
from inference import inference
from dataloader import GQADataset#, VisualGenomeDataset, SNLIVEDataset, CLEVRobotDataset, AI2THORDataset

import sys
sys.path.append('./Grounded-Segment-Anything')


if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # # Command-line argument parsing
    # parser = argparse.ArgumentParser(description='Command line arguments')
    # parser.add_argument('--run_mode', type=str, default=None, help='Override run_mode (train, eval, prepare_cs, train_cs, eval_cs)')
    # cmd_args = parser.parse_args()

    # # Override args from config.yaml with command-line arguments if provided
    # args['training']['run_mode'] = cmd_args.run_mode if cmd_args.run_mode is not None else args['training']['run_mode']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Prepare datasets
    print("Loading the datasets...")
    test_dataset = GQADataset(args)

    torch.manual_seed(0)
    if args['datasets']['use_num_test_data']:
        test_subset_idx = torch.randperm(len(test_dataset))[:int(args['datasets']['num_test_data'])]
    else:
        test_subset_idx = torch.randperm(len(test_dataset))[:int(args['datasets']['percent_test'] * len(test_dataset))]
    test_subset = Subset(test_dataset, test_subset_idx)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    print('num of train, test:', 0, len(test_subset))

    # Start inference
    print(args)
    inference(device, args, test_loader)
