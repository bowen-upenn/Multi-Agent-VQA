import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os

import sys
sys.path.append('./Grounded-Segment-Anything')

from grounded_sam_interface import query_grounded_sam, query_grounding_dino
from groundingdino.util.inference import Model as DINO

from utils import *
from query_vlm import QueryVLM
from query_llm import QueryLLM


def inference(gpu, args, test_subset):
    rank = gpu
    world_size = torch.cuda.device_count()
    setup(rank, world_size)
    print('rank', rank, 'torch.distributed.is_initialized', torch.distributed.is_initialized())

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0, drop_last=True, sampler=test_sampler)

    # Building GroundingDINO inference model
    grounding_dino_model = DINO(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    grounding_dino_model = DDP(grounding_dino_model).to(rank)
    grounding_dino_model.eval()

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            try:
                image_id, image_path, question, answer = data['image_id'], data['image_path'], data['question'], data['answer']
            except:
                continue

            # extract related object instances from the task prompt
            print('question', question)
            related_objects = QueryLLM.query_llm(prompts, llm_model=args['llm']['llm_model'], step='related_objects')
            print('related_objects', related_objects)

            # query grounded sam on the input image 
            boxes, logits, phrases = query_grounding_dino(rank, args, grounding_dino_model, image_path, text_prompt=related_objects)
            print('boxes', boxes.shape, 'logits', logits.shape, 'phrases', phrases)

            # find all object instances in the scene

            # find object instances related to the task prompt


            # query a large vision-language agent on the attributes of each object instance


            # merge attributes and class labels of all objects as a system prompt


            # query another a large vision-language agent on relation predictions and complete downstream tasks





