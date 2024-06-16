import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils import *


class GQADataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset_split = args['datasets']['gqa_dataset_split']

        if self.dataset_split == 'val':
            self.questions_file = self.args['datasets']['gqa_val_questions_file']
        elif self.dataset_split == 'val-subset':
            self.questions_file = self.args['datasets']['gqa_val_subset_questions_file']
        else:
            self.questions_file = self.args['datasets']['gqa_test_questions_file']

        self.images_dir = self.args['datasets']['gqa_images_dir']
        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if self.dataset_split == 'val-subset':
            annot = self.questions[idx]
            image_id = annot['image']
            image_path = os.path.join(self.images_dir, image_id)
        else:
            annot = self.questions[list(self.questions.keys())[idx]]
            image_id = annot['imageId']
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        question = annot['question']
        answer = annot['answer']

        if self.args['inference']['verbose']:
            curr_data = 'image_path: ' + image_path + ' question: ' + question + ' answer: ' + answer
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'question_id': -1, 'answer': answer}


class VQAv2Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset_split = args['datasets']['vqa_v2_dataset_split']
        self.transform = transform

        if self.dataset_split == 'val':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_val_questions_file']
            self.answers_file = self.args['datasets']['vqa_v2_val_annotations_file']
            with open(self.answers_file, 'r') as f:
                self.answers = json.load(f)
                self.answers = self.answers['annotations']    # it is a list of dictionaries

        elif self.dataset_split == 'val1000':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_val1000_questions_file']
            self.answers_file = self.args['datasets']['vqa_v2_val1000_annotations_file']
            with open(self.answers_file, 'r') as f:
                self.answers = json.load(f)
                self.answers = self.answers['annotations']

        if self.dataset_split == 'rest-val':
            self.images_dir = self.args['datasets']['vqa_v2_val_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_rest_val_questions_file']
            self.answers_file = self.args['datasets']['vqa_v2_rest_val_annotations_file']
            with open(self.answers_file, 'r') as f:
                self.answers = json.load(f)

        elif self.dataset_split == 'test':
            self.images_dir = self.args['datasets']['vqa_v2_test_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_test_questions_file']
            self.answers_file = None

        elif self.dataset_split == 'test-dev':
            self.images_dir = self.args['datasets']['vqa_v2_test_images_dir']
            self.questions_file = self.args['datasets']['vqa_v2_test_dev_questions_file']
            self.answers_file = None

        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)
            self.questions = self.questions['questions']    # it is a list of dictionaries

        # self.questions = [annot for annot in self.questions if annot['question_id'] == 489588003]
        # print(f'len(self.questions): {len(self.questions)}', self.questions)

    def __len__(self):
        # VQA-v2 test-dev 107394
        return len(self.questions)

    def __getitem__(self, idx):
        annot = self.questions[idx]

        image_id = annot['image_id']
        if self.dataset_split == 'val' or self.dataset_split == 'val1000' or self.dataset_split == 'rest-val':
            image_path = os.path.join(self.images_dir, f"COCO_val2014_{image_id:012}.jpg")
        else:
            image_path = os.path.join(self.images_dir, f"COCO_test2015_{image_id:012}.jpg")

        question = annot['question']
        question_id = annot['question_id']

        if self.answers_file is not None:
            if self.dataset_split == 'rest-val':    # question_id as the dictionary key
                answer = self.answers[str(question_id)]['multiple_choice_answer']
            else:
                answer = self.answers[idx]['multiple_choice_answer']    # same idx order as questions
        else:
            answer = ""

        if self.args['inference']['verbose']:
            curr_data = 'image_id: ' + str(image_id) + ' image_path: ' + image_path + ' question: ' + question + \
                        ' question_id: ' + str(question_id) + ' answer: ' + answer
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'question_id': question_id, 'answer': answer}


class VQASyntheticDataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args

        # UPDATE vqa_synthetic_val_questions_file, 
        # vqa_synthetic_test_questions_dir, vqa_synthetic_images_dir in config.yaml,
        # and vqa_synthetic_answer_dir
        self.questions_file = self.args['datasets']['synthetic_data_gpt4_dalle3_questions_file']
        self.images_dir = self.args['datasets']['synthetic_data_gpt4_dalle3_images_dir']
        if 'synthetic_data_gpt4_dalle3_answer_file' in self.args['datasets']:
            self.answer_file = self.args['datasets']['synthetic_data_gpt4_dalle3_answer_file']
        else:
            raise KeyError("Key 'synthetic_data_gpt4_dalle3_answer_file' not found in the dataset configuration.")
        self.description_dir = self.args['datasets']['synthetic_data_gpt4_dalle3_description_dir']

        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)
            self.questions = self.questions['questions']
        
        with open(self.answer_file, 'r') as f:
            self.answers = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        annot = self.questions[idx]

        image_id = annot['image_id']
        image_path = os.path.join(self.images_dir, f"image_{image_id}.png")
        # error check: image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image path {image_path} does not exist.")
        question = annot['question']
        question_id = annot['question_id']
        answer = self.answers[str(question_id)]

        if self.args['inference']['verbose']:
            curr_data = 'image_path: ' + image_path + ' question: ' + question + ' answer: ' + answer
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'question_id': question_id, 'answer': answer}

##########################################################
## TODO: Add Clever and A-OKVQA datasets
class CLEVRDataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset_split = args['datasets']['clevr_dataset_split']

        # UPDATE clevr_val_questions_file, clevr_test_questions_file, and clevr_images_dir in config.yaml
        if self.dataset_split == 'val':
            self.questions_file = self.args['datasets']['clevr_val_questions_file']
        else:
            self.questions_file = self.args['datasets']['clevr_test_questions_file']

        self.images_dir = self.args['datasets']['clevr_images_dir']
        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        if self.args['inference']['verbose']:
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {}


class AOKVQADataset(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset_split = args['datasets']['a_okvqa_dataset_split']

        # UPDATE a_okvqa_val_questions_file, a_okvqa_test_questions_file, and a_okvqa_images_dir in config.yaml
        if self.dataset_split == 'val':
            self.questions_file = self.args['datasets']['a_okvqa_val_questions_file']
        else:
            self.questions_file = self.args['datasets']['a_okvqa_test_questions_file']

        self.images_dir = self.args['datasets']['a_okvqa_images_dir']
        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        if self.args['inference']['verbose']:
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {}
