from tqdm import tqdm
import os

import sys
sys.path.append('./Grounded-Segment-Anything')

from detections import query_sam, query_grounded_sam, query_grounding_dino
from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from utils import *
from query_vlm import QueryVLM
from query_llm import QueryLLM


def inference(device, args, test_loader):
    # Building GroundingDINO inference model
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    sam = sam_model_registry[args['sam']['SAM_ENCODER_VERSION']](checkpoint=args['sam']['SAM_CHECKPOINT_PATH']).to(device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam)
    LLM, VLM = QueryLLM(), QueryVLM()

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            print('data', data)
            image_id, image_path, question, answer = data['image_id'], data['image_path'], data['question'], data['answer']

            # extract related object instances from the task prompt
            print('question', question)
            related_objects = LLM.query_llm(question, llm_model=args['llm']['llm_model'], step='related_objects')
            print('related_objects', related_objects)

            # query grounded sam on the input image
            # the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
            # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
            image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=related_objects)
            print('boxes', boxes.shape, 'logits', logits.shape, 'phrases', phrases, 'image', image.shape)

            # find all object instances in the scene
            masks = query_sam(device, args, sam_mask_generator, image)
            print('masks', len(masks))

            # find object instances related to the task prompt


            # query a large vision-language agent on the attributes of each object instance


            # merge attributes and class labels of all objects as a system prompt


            # query another a large vision-language agent on relation predictions and complete downstream tasks





