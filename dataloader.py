import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class GQADataset(Dataset):
    def __init__(self, args, transform=None):
        """
        Args:
            questions_file (string): Path to the JSON file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.args = args
        self.questions_file = self.args['datasets']['gqa_questions_file']
        self.images_dir = self.args['datasets']['gqa_images_dir']
        self.transform = transform
        with open(self.questions_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        annot = self.questions[list(self.questions.keys())[idx]]

        image_id = annot['imageId']
        image_path = os.path.join(self.images_dir, f"{annot['imageId']}.jpg")
        question = annot['question']
        answer = annot['answer']
        print('image_path:', image_path, 'question:', question, 'answer:', answer)

        return {'image_id': image_id, 'image_path': image_path, 'question': question, 'answer': answer}