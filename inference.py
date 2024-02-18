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

    submit_outputs_vqa = False
    if args['datasets']['dataset'] == 'vqa-v2' and (args['datasets']['vqa_v2_dataset_split'] == 'test' or args['datasets']['vqa_v2_dataset_split'] == 'test-dev'):
        submit_outputs_vqa = True
        answer_list = load_answer_list(args['datasets']['vqa_v2_answer_list'])
        # if os.path.exists('outputs/submit_vqav2_' + args['datasets']['vqa_v2_dataset_split'] + '_2.json'):
        #     os.remove('outputs/submit_vqav2_' + args['datasets']['vqa_v2_dataset_split'] + '_2.json')
        if os.path.exists('outputs/responses.json'):
            os.remove('outputs/responses.json')

    with torch.no_grad():
        for batch_count, data in enumerate(tqdm(test_loader), 0):
            image_id, image_path, question, question_id, target_answer = data['image_id'], data['image_path'], data['question'], data['question_id'], data['answer']
            assert len(image_path) == 1

            image = np.asarray(Image.open(image_path[0]).convert("RGB"))

            # try to answer the visual question using the baseline large VLM model directly without calling multi-agents
            answer = VLM.query_vlm(image, question[0], step='ask_directly', verbose=args['inference']['verbose'])[0]

            # if the answer failed, reattempt the visual question answering task with additional information assisted by the object detection model
            match_baseline_failed = re.search(r'\[Answer Failed\]', answer) is not None or re.search(r'sorry', answer.lower()) is not None or \
                                    re.search(r'\[Numeric Answer Needs Further Assistance\]', answer) is not None or len(answer) == 0
            verify_numeric_answer = args['datasets']['dataset'] == 'vqa-v2' and re.search(r'\[Answer Needs Further Assistance\]', answer) is not None

            if match_baseline_failed:
                if args['inference']['verbose']:
                    msg = "The baseline model failed to answer the question or needed further assistance to verify a numeric answer. Reattempting with multi-agents."
                    print(f'{Colors.WARNING}{msg}{Colors.ENDC}')

                # extract object instances needed to solve the visual question answering task
                needed_objects = LLM.query_llm(question, previous_response=answer, llm_model=args['llm']['llm_model'], step='needed_objects',
                                               verify_numeric_answer=verify_numeric_answer, verbose=args['inference']['verbose'])

                # query grounded sam on the input image
                # the 'boxes' is a tensor of shape (N, 4) where N is the number of object instances in the image,
                # the 'logits' is a tensor of shape (N), and the 'phrases' is a list of length (N) such as ['table', 'door']
                image, boxes, logits, phrases = query_grounding_dino(device, args, grounding_dino, image_path[0], text_prompt=needed_objects)

                # query a large vision-language agent on the attributes of each object instance
                object_attributes = VLM.query_vlm(image, question[0], step='attributes', phrases=phrases, bboxes=boxes, verbose=args['inference']['verbose'])

                # merge object descriptions as a system prompt and reattempt the visual question answering
                reattempt_answer = VLM.query_vlm(image, question[0], step='reattempt', obj_descriptions=object_attributes[0], prev_answer=answer,
                                                 verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=args['inference']['verbose'])[0]

                if submit_outputs_vqa:
                    save_output_predictions_vqav2(question_id, reattempt_answer, answer_list, split=args['datasets']['vqa_v2_dataset_split'], verbose=args['inference']['verbose'])
                else:
                    # grade the answer. vqa-v2 test and test-dev datasets do not have ground truth answers available
                    grades = []
                    for grader_id in range(3):
                        grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=reattempt_answer, step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

                # record responses to json file
                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer':verify_numeric_answer, 'initial_answer': answer, 'reattempt_answer': reattempt_answer,
                                 'needed_objects': needed_objects, 'object_attributes': object_attributes[0], 'boxes': str(boxes), 'logits': str(logits), 'phrases': phrases, 'grades': grades}

            else:
                if submit_outputs_vqa:
                    save_output_predictions_vqav2(question_id, answer, answer_list, split=args['datasets']['vqa_v2_dataset_split'], verbose=args['inference']['verbose'])
                else:
                    grades = []
                    for grader_id in range(3):
                        grades.append(LLM.query_llm(question, target_answer=target_answer[0], model_answer=answer, step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

                # record responses to json file
                response_dict = {'image_id': str(image_id[0].item()), 'image_path': image_path[0], 'question_id': str(question_id[0].item()), 'question': question[0], 'target_answer': target_answer[0],
                                 'match_baseline_failed': match_baseline_failed, 'verify_numeric_answer': verify_numeric_answer, 'initial_answer': answer, 'grades': grades}

            if not submit_outputs_vqa:
                majority_vote = accumulate_grades(args, grader, grades, match_baseline_failed)
                response_dict['majority_vote'] = majority_vote

                if (batch_count + 1) % args['inference']['print_every'] == 0:
                    accuracy = grader.average_score()
                    print('Accuracy at batch idx ', batch_count, '(baseline, final)', accuracy)

            write_response_to_json(question_id, response_dict)

        if not submit_outputs_vqa:
            accuracy = grader.average_score()
            record_final_accuracy(accuracy)
            print('Accuracy (baseline, final)', accuracy)
