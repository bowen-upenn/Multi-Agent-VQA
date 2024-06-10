import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import yaml
import os
import json
import argparse
import vertexai

from utils import *
from inference import inference
from dataloader import * #, VisualGenomeDataset, SNLIVEDataset, CLEVRobotDataset, AI2THORDataset

import sys
sys.path.append('./Grounded-Segment-Anything')
sys.path.append('./CLIP_Count')

########## Environment Variable for Gemini Pro Vision #############
credential_path = "/raid0/docker-raid/bwjiang/vlm4sgg/LLM_api_keys/multi-agent-vqa-gemini-eb6d477d5c97.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
########## Environment Variable for Gemini Pro Vision #############

if __name__ == "__main__":
    print('Torch', torch.__version__, 'Torchvision', torchvision.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--vlm_model', type=str, default="gpt4", help='Set VLM model (gpt4, gemini)')
    parser.add_argument('--dataset', type=str, default=None, help='Set dataset (gqa, vqa-v2, clevr, a-okvqa)')
    parser.add_argument('--split', type=str, default=None, help='Set dataset split. gqa: val, val-subset, test. '
                                                                'vqa-v2: val, rest-val, val1000, test-dev, test-std.'
                                                                'clevr: val, test.'
                                                                'a-okvqa: val, test.')
    parser.add_argument('--prompt', type=str, default='baseline', help='Set prompt type (baseline, simple, cot, ps)')
    parser.add_argument('--multi_agent', dest='multi_agent', action='store_true', help='Use multi-agent pipeline. prompt_type should not be baseline.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    # Override args from config.yaml with command-line arguments if provided
    args['vlm']['vlm_model'] = cmd_args.vlm_model
    if re.search(r'gemini', args['vlm']['vlm_model']) is not None:
        print("Using Gemini Pro Vision as VLM, initializing the Google Cloud Certificate")
        credential_path = "LLM_api_keys/multi-agent-vqa-gemini-eb6d477d5c97.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        PROJECT_ID = "multi-agent-vqa-gemini"
        REGION = "us-central1"
        vertexai.init(project=PROJECT_ID, location=REGION)
    args['datasets']['dataset'] = cmd_args.dataset if cmd_args.dataset is not None else args['datasets']['dataset']
    args['inference']['prompt_type'] = cmd_args.prompt if cmd_args.prompt is not None else args['inference']['prompt_type']
    args['inference']['multi_agent'] = cmd_args.multi_agent if cmd_args.multi_agent is not None else args['inference']['multi_agent']
    # if args['inference']['prompt_type'] == 'baseline':
    #     args['inference']['multi_agent'] = False
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']
    if args['datasets']['dataset'] == 'gqa':
        args['datasets']['gqa_dataset_split'] = cmd_args.split if cmd_args.split is not None else args['datasets']['gqa_dataset_split']
    elif args['datasets']['dataset'] == 'vqa-v2':
        args['datasets']['vqa_v2_dataset_split'] = cmd_args.split if cmd_args.split is not None else args['datasets']['vqa_v2_dataset_split']
    elif args['datasets']['dataset'] == 'clevr':
        args['datasets']['clevr_dataset_split'] = cmd_args.split if cmd_args.split is not None else args['datasets']['clevr_dataset_split']
    elif args['datasets']['dataset'] == 'a-okvqa':
        args['datasets']['a_okvqa_dataset_split'] = cmd_args.split if cmd_args.split is not None else args['datasets']['a_okvqa_dataset_split']
    else:
        raise ValueError('Invalid dataset name')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Prepare datasets
    print("Loading the datasets...")
    if args['datasets']['dataset'] == 'gqa':
        test_dataset = GQADataset(args)
    elif args['datasets']['dataset'] == 'vqa-v2':
        test_dataset = VQAv2Dataset(args)
    elif args['datasets']['dataset'] == 'clevr':
        test_dataset = CLEVRDataset(args)
    elif args['datasets']['dataset'] == 'a-okvqa':
        test_dataset = AOKVQADataset(args)
    else:
        raise ValueError('Invalid dataset name')

    torch.manual_seed(0)
    # [zhijunz] TEMP, set for debug
    # args['datasets']['use_num_test_data'] = 3
    if args['datasets']['use_num_test_data']:
        test_subset_idx = torch.randperm(len(test_dataset))[:int(args['datasets']['num_test_data'])]
    else:
        # test_subset_idx = torch.randperm(len(test_dataset))[3600:]
        test_subset_idx = torch.randperm(len(test_dataset))[:int(args['datasets']['percent_test'] * len(test_dataset))]
    test_subset = Subset(test_dataset, test_subset_idx)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    print('num of train, test:', 0, len(test_subset))

    # Start inference
    print(args)
    inference(device, args, test_loader)
