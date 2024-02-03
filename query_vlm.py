import torch
import numpy as np
from tqdm import tqdm
import os
import json
import openai
# from openai import OpenAI
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


    def get_image(self, image_path, bbox=None):
        if image_path not in self.image_cache:
            image = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(self.image_size, self.image_size))

            if bbox is not None:
                x1, x2, y1, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                image = image[y1:y2, x1:x2]

            # Convert image to bytes for base64 encoding
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = np.array(buffer).tobytes()

            self.image_cache[image_path] = base64.b64encode(image_bytes).decode('utf-8')
        return self.image_cache[image_path]


    def query_vlm(self, image_paths, prompt, max_batch_size=4): # "Describe the attributes and the name of the object in the image"
        # query on a single image
        if len(image_paths) == 1:
            response = self._query_openai_gpt_4v(image_path[0], prompt)
            return response

        # query on a batch of images in parallel
        responses = []
        total_num_images = len(image_paths)

        # process images in batches to avoid exceeding OpenAI API limits
        for start_idx in range(0, total_num_images, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_num_images)
            batch_image_paths = image_paths[start_idx:end_idx]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_batch_size) as executor:
                batch_responses = list(executor.map(lambda image_path: self._query_openai_gpt_4v(image_path, prompt), batch_image_paths))
            responses.extend(batch_responses)

        return responses


    def _query_openai_gpt_4v(self, image_path, prompt, verbose=False):
            # Load and process image and annotations if they exist
        if os.path.exists(image_path):
            image_path = os.path.join(image_dir, annot_name[:-16] + '.jpg')
            if verbose:
                print('annot_name', annot_name, 'image_path', image_path, 'edge', edge)

            base64_image = self.image_cache.get_image(image_path)

            # Form the prompt including the image.
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            prompt = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 300
            }

            # Send request to OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
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
        else:
            print(f'Image not found!')
            return ""
