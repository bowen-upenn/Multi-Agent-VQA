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

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'answer': answer}


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
            print('answers', self.answers[0])

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

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        annot = self.questions[idx]

        image_id = annot['image_id']
        if self.dataset_split == 'val':
            image_path = os.path.join(self.images_dir, f"COCO_val2014_{image_id:012}.jpg")
        else:
            image_path = os.path.join(self.images_dir, f"COCO_test2015_{image_id:012}.jpg")

        question = annot['question']
        if self.answers_file is not None:
            answer = self.answers[idx]['multiple_choice_answer']
        else:
            answer = ""

        if self.args['inference']['verbose']:
            curr_data = 'image_path: ' + image_path + ' question: ' + question + ' answer: ' + answer
            print(f'{Colors.HEADER}{curr_data}{Colors.ENDC}')

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'answer': answer}