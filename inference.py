from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import re

from detections import query_sam, query_grounded_sam, query_grounding_dino
from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from utils import *
from query_vlm import QueryVLM
from query_llm import QueryLLM


def inference(device, args, test_loader):
    # Building GroundingDINO inference model
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    LLM, VLM = QueryLLM(args), QueryVLM(args)
    grader = Grader()

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, target_answer = data['image_id'], data['image_path'], data['question'], data['answer']
            assert len(image_path) == 1

            image = np.asarray(Image.open(image_path[0]).convert("RGB"))

            match_baseline_failed = False
            if not args['inference']['force_multi_agents']:
                # try to answer the visual question using the baseline large VLM model directly without calling multi-agents
                answer = VLM.query_vlm(image, question[0], step='ask_directly', verbose=args['inference']['verbose'])[0]

                # if the answer failed, reattempt the visual question answering task with additional information assisted by the object detection model
                match_baseline_failed = re.search(r'\[Answer Failed\]', answer) or re.search(r'sorry', answer.lower()) or len(answer) == 0

            if match_baseline_failed or args['inference']['force_multi_agents']:
                msg = "The baseline model failed to answer the question. Reattempting with additional information via multi-agents."
                print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                # extract object instances needed to solve the visual question answering task
                needed_objects = LLM.query_llm(question, previous_response=answer, llm_model=args['llm']['llm_model'], step='needed_objects', verbose=args['inference']['verbose'])

                # query grounded sam on the input image
                # the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
                # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
                image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)

                # query a large vision-language agent on the attributes of each object instance
                object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=args['inference']['verbose'])

                # merge object descriptions as a system prompt and reattempt the visual question answering
                reattempt_answer = VLM.query_vlm(image, question[0], step='relations', obj_descriptions=object_attributes[0], prev_answer=answer, verbose=args['inference']['verbose'])

                # grade the answer
                grades = []
                for grader_id in range(3):
                    grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=reattempt_answer[0], step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))
            else:
                grades = []
                for grader_id in range(3):
                    grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=answer, step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

            accumulate_grades(args, grader, grades, match_baseline_failed)
            if (batch_count + 1) % args['inference']['print_every'] == 0:
                accuracy = grader.average_score()
                print('Accuracy at batch idx ', batch_count, '(baseline, final)', accuracy)

        accuracy = grader.average_score()
        print('Accuracy (baseline, final)', accuracy)
