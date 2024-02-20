from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import re

from groundingdino.util.inference import load_model
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from CLIP_Count.run import Model as CLIP_Count

from query_vlm import QueryVLM
from query_llm import QueryLLM
from detections import query_sam, query_grounded_sam, query_grounding_dino
from counting import query_clip_count
from utils import *


def inference(device, args, test_loader):
    # Building GroundingDINO, LLM, VLM, and CLIP_Count models as multi-agents
    grounding_dino = load_model(args['dino']['GROUNDING_DINO_CONFIG_PATH'], args['dino']['GROUNDING_DINO_CHECKPOINT_PATH'])
    LLM, VLM = QueryLLM(args), QueryVLM(args)
    clip_count = CLIP_Count.load_from_checkpoint('CLIP_Count/ckpt/clipcount_pretrained.ckpt', strict=False).to(device)
    clip_count.eval()  # Set the model to evaluation mode

    grader = Grader()
    output_response_filename = args['inference']['output_response_filename']

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, question_id, target_answer = data['image_id'], data['image_path'], data['question'], data['question_id'], data['answer']
            assert len(image_path) == 1

            image = np.asarray(Image.open(image_path[0]).convert("RGB"))

            # """ There could be three cases at the beginning
            # 1. The question is about counting, i.e., [Numeric Answer]: verify_numeric_answer is True and match_baseline_failed is False
            # 2. The question is about counting, but the model failed to find any object to be counted or the number maybe zero, i.e., [Answer Failed]: verify_numeric_answer is False and match_baseline_failed is True
            # 3. The question is not about counting, i.e., [Not Numeric Answer]: verify_numeric_answer is False and match_baseline_failed is False
            # """
            # # check if the answer should be a numeric answer
            # answer = VLM.query_vlm(image, question[0], step='check_numeric_answer', verbose=args['inference']['verbose'])[0]
            # verify_numeric_answer = args['datasets']['dataset'] == 'vqa-v2' and re.search(r'\[Numeric Answer\]', answer) is not None
            # # if the model thinks the numeric answer is zero, i.e., it couldn't find any object to be counted, it will return a failed answer
            # match_baseline_failed = re.search(r'\[Answer Failed\]', answer) is not None or re.search(r'sorry', answer.lower()) is not None or len(answer) == 0
            # if match_baseline_failed:
            #     verify_numeric_answer = False   # ensure that the model does not output both [Numeric Answer] and [Answer Failed] at the same time
            #
            # # if the question is not about counting, first try to answer the visual question using the baseline large VLM model directly without calling multi-agents
            # if not verify_numeric_answer and not match_baseline_failed:
            answer = VLM.query_vlm(image, question[0], step='ask_directly', verbose=args['inference']['verbose'])[0]

            # if the answer failed, reattempt the visual question answering task with additional information assisted by the object detection model
            match_baseline_failed = re.search(r'\[Answer Failed\]', answer) is not None or re.search(r'sorry', answer.lower()) is not None or len(answer) == 0
            verify_numeric_answer = re.search(r'\[Non-zero Numeric Answer\]', answer) is not None

            if match_baseline_failed:
                if args['inference']['verbose']:
                    if verify_numeric_answer:
                        msg = "The baseline model needs further assistance to predict a numeric answer. Reattempting with multi-agents."
                    else:
                        msg = "The baseline model failed to answer the question initially with missing objects. Reattempting with multi-agents."
                    print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                # extract object instances needed to solve the visual question answering task
                needed_objects = LLM.query_llm(question, previous_response=answer, llm_model=args['llm']['llm_model'], step='needed_objects',
                                               verify_numeric_answer=verify_numeric_answer, verbose=args['inference']['verbose'])

                if verify_numeric_answer:
                    reattempt_answer = query_clip_count(device, image, clip_count, prompts=needed_objects, verbose=args['inference']['verbose'])
                else:
                    # query grounded sam on the input image. the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
                    # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
                    image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)

                    # query a large vision-language agent on the attributes of each object instance
                    object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=args['inference']['verbose'])

                    # merge object descriptions as a system prompt and reattempt the visual question answering
                    reattempt_answer = VLM.query_vlm(image, question[0], step='reattempt', obj_descriptions=object_attributes[0], prev_answer=answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=args['inference']['verbose'])[0]

                # grade the answer. vqa-v2 test and test-dev datasets do not have ground truth answers available
                grades = []
                for grader_id in range(3):
                    grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=reattempt_answer, step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

                # record responses to json file
                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer': verify_numeric_answer, 'initial_answer': answer, 'reattempt_answer': reattempt_answer,
                                 'needed_objects': needed_objects, 'grades': grades}
                if not verify_numeric_answer:
                    response_dict['object_attributes'] = object_attributes[0]
                    response_dict['boxes'] = str(boxes)
                    response_dict['logits'] = str(logits)
                    response_dict['phrases'] = phrases

            else:
                grades = []
                for grader_id in range(3):
                    grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=answer, step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

                # record responses to json file
                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer': verify_numeric_answer, 'initial_answer': answer, 'grades': grades}

            majority_vote = accumulate_grades(args, grader, grades, match_baseline_failed)
            response_dict['majority_vote'] = majority_vote

            if (batch_count + 1) % args['inference']['print_every'] == 0:
                baseline_accuracy, final_accuracy, _ = grader.average_score()
                print('Accuracy at batch idx ', batch_count, '(baseline, final)', baseline_accuracy, final_accuracy)

            if args['inference']['save_output_response']:
                write_response_to_json(question_id, response_dict, output_response_filename)

        baseline_accuracy, final_accuracy, stats = grader.average_score()
        if args['inference']['save_output_response']:
            record_final_accuracy(baseline_accuracy, final_accuracy, stats, output_response_filename)
        print('Accuracy (baseline, final)', baseline_accuracy, final_accuracy, 'stats', stats)
