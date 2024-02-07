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
from torchvision.ops import box_convert


class QueryVLM:
    def __init__(self, image_size=512):
        self.image_cache = {}
        self.image_size = image_size

        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()


    def process_image(self, image, bbox=None):
        # we have to crop the image before converting it to base64
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            print("bbox: ", bbox)
            xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy")
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.imwrite('test_images/original_image' + str(bbox) + '.jpg', image)

            # increase the receptive field of each box to include possible nearby objects and contexts
            image = image[y1:y2, x1:x2]
            cv2.imwrite('test_images/cropped_image' + str(bbox) + '.jpg', image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()
        image_bytes = base64.b64encode(image_bytes).decode('utf-8')

        return image_bytes


    def messages_to_query_object_attributes(self, phrase=None):
        if phrase is not None:
            return "Describe the attributes and the name of the object in the image in one or two sentences, " \
                   "with the focus on the " + phrase + " and nearby objects, " \
                   "including visual attributes like color, shape, size, materials, and clothes if the object is a person, " \
                   "and semantic attributes like type and current status if applicable."
        else:
            return "Describe the attributes and the name of the objects in the image in one or two sentences, " \
                   "including visual attributes like color, shape, size, and materials, and clothes if the object is a person, " \
                   "and semantic attributes like type and current status if applicable."


    def query_vlm(self, image, phrases=None, step='attributes', bboxes=None): # "Describe the attributes and the name of the object in the image"
        if len(bboxes) == 0 or bboxes is None:
            response = self._query_openai_gpt_4v(image, step)
            return response

        # query on a single image
        if len(bboxes) == 1:
            bbox = bboxes.squeeze(0)
            phrase = phrases[0]
            response = self._query_openai_gpt_4v(image, step, phrase, bbox)
            return response

        # query on a batch of images in parallel
        responses = []
        total_num_objects = len(bboxes)

        # process all objects from the same image in a parallel batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
            batch_responses = list(executor.map(lambda bbox, phrase: self._query_openai_gpt_4v(image, step, phrase, bbox), bboxes, phrases))
        responses.append(batch_responses)

        return responses


    def _query_openai_gpt_4v(self, image, step, phrase=None, bbox=None, verbose=True):
        # we have to crop the image before converting it to base64
        base64_image = self.process_image(image, bbox)

        if step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes()
            else:
                messages = self.messages_to_query_object_attributes(phrase)
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
