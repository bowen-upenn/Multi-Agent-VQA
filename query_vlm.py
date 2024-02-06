import torch
import numpy as np
from tqdm import tqdm
import os
import json
import random
import cv2
import base64
import requests
import concurrent.futures


class QueryVLM:
    def __init__(self, image_size=512):
        self.image_cache = {}
        self.image_size = image_size

        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()


    def get_image(self, image_path, bbox=None): # bbox = [X, Y, W, H]
        if image_path not in self.image_cache:
            image = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))

            # Convert image to bytes for base64 encoding
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = np.array(buffer).tobytes()
            self.image_cache[image_path] = base64.b64encode(image_bytes).decode('utf-8')
        else:
            image_bytes = self.image_cache[image_path]

        if bbox is not None:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            image_bytes = image_bytes[y:y+h, x:x+w]

        return image_bytes


    def process_image(self, image, bbox=None):
        # we have to crop the image before converting it to base64
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            print('bbox', bbox)
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.imwrite('original_image' + str(bbox) + '.jpg', image)
            image = image[y:y+h, x:x+w]
            cv2.imwrite('cropped_image' + str(bbox) + '.jpg', image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()
        image_bytes = base64.b64encode(image_bytes).decode('utf-8')

        return image_bytes


    def messages_to_query_object_attributes(self):
        return "Describe the attributes and the name of the object in the image, including its visual attributes like color, shape, size, and materials," \
               "and semantic attributes like type and current status if applicable."


    def query_vlm(self, image, step='attributes', bboxes=None): # "Describe the attributes and the name of the object in the image"
        # query on a single image
        if len(bboxes) == 1 or bboxes is None:
            bbox = bboxes.squeeze(0)
            response = self._query_openai_gpt_4v(image, step, bbox)
            return response

        # query on a batch of images in parallel
        responses = []
        total_num_objects = len(bboxes)

        # process all objects from the same image in a parallel batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
            batch_responses = list(executor.map(lambda bbox: self._query_openai_gpt_4v(image, step, bbox), bboxes))
        responses.append(batch_responses)

        return responses


    def _query_openai_gpt_4v(self, image, step, bbox=None, verbose=True):
        # we have to crop the image before converting it to base64
        base64_image = self.process_image(image, bbox)

        if step == 'attributes':
            messages = self.messages_to_query_object_attributes()
            max_tokens = 200

        # Form the prompt including the image.
        # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
        prompt = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": messages},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": max_tokens
        }

        # Send request to OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=prompt)
        response_json = response.json()

        # Process the response
        # Check if the response is valid and contains the expected data
        if 'choices' in response_json and len(response_json['choices']) > 0:
            completion_text = response_json['choices'][0].get('message', {}).get('content', '')

            if verbose:
                print(f'Response: {completion_text}')

            return completion_text
        else:
            return ""
