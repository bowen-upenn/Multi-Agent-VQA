from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import re

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
    sam_mask_generator = SamAutomaticMaskGenerator(sam,
                                                   min_mask_region_area=args['sam']['min_mask_region_area'],
                                                   pred_iou_thresh=args['sam']['pred_iou_thresh'],
                                                   stability_score_thresh=args['sam']['stability_score_thresh'])
    LLM, VLM = QueryLLM(args), QueryVLM(args)

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, answer = data['image_id'], data['image_path'], data['question'], data['answer']
            assert len(image_path) == 1

            print('question', question)
            image = np.asarray(Image.open(image_path[0]).convert("RGB"))
            answer = VLM.query_vlm(image, question[0], step='ask_directly')[0]

            # find if the answer failed or not
            pattern = r'\[Answer Failed\]'
            match = re.search(pattern, answer)
            if match:
                # extract object instances needed to solve the question
                needed_objects = LLM.query_llm(question, previous_response=answer, llm_model=args['llm']['llm_model'], step='needed_objects')

                # query grounded sam on the input image
                # the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
                # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
                image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)
                print('boxes', boxes.shape, 'logits', logits.shape, 'phrases', phrases, 'image', image.shape)

                # query a large vision-language agent on the attributes of each object instance
                object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes)

                # merge object descriptions as a system prompt
                reattempt_answer = VLM.query_vlm(image, question[0], step='relations', obj_descriptions=object_attributes, prev_answer=answer)


            # extract related object instances from the task prompt
            # related_objects = LLM.query_llm(question, llm_model=args['llm']['llm_model'], step='related_objects')
            # print('related_objects', related_objects)
            #
            # # query grounded sam on the input image
            # # the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
            # # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
            # image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=related_objects)
            # print('boxes', boxes.shape, 'logits', logits.shape, 'phrases', phrases, 'image', image.shape)
            #
            # # if args['inference']['find_nearby_objects']:
            # #     # find all object instances in the scene
            # #     masks = query_sam(device, args, sam_mask_generator, image)
            # #     nearby_boxes = torch.tensor([mask['bbox'] for mask in masks])
            # #     print('masks', len(masks), masks[0].keys(), masks[0]['bbox'], 'nearby_boxes', nearby_boxes.shape)
            # #
            # #     nearby_boxes = filter_boxes_pytorch(boxes, nearby_boxes, args['inference']['nearby_bbox_iou_threshold'])
            # #     print('nearby_boxes', nearby_boxes.shape)
            #
            # # query a large vision-language agent on the attributes of each object instance
            # object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes)
            #
            # # merge object descriptions as a system prompt
            # answers = VLM.query_vlm(image, question[0], step='relations', obj_descriptions=object_attributes)


            # query another a large vision-language agent on relation predictions and complete downstream tasks





