import torch
import numpy as np
from tqdm import tqdm
import os
import json
# import openai
from openai import OpenAI
import random
import cv2


class QueryLLM:
    def __init__(self):
        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()


    def messages_to_extract_objects_of_interest(self, prompt):
        messages = [
            {"role": "system", "content": "You are a language assistant."},
            {"role": "user", "content": "Please extract all objects mentioned in the following sentence, and separate each object with '.': "
                                        "Is there a red apple on the table?"},
            {"role": "assistant", "content": "Red apple . Table ."},
            {"role": "user", "content": "Please extract all objects mentioned in the following sentence, and separate each object with '.': " + prompt}
        ]
        return messages


    def query_llm(self, prompts, max_batch_size=4, llm_model='gpt-3.5-turbo', step='related_objects'):
        # query on a single image
        if len(prompts) == 1:
            if llm_model == 'gpt-4':
                response = self._query_openai_gpt_4(prompts[0], step)
            else:
                response = self._query_openai_gpt_3p5(prompts[0], step)
            return response

        # query on a batch of images in parallel
        responses = []
        total_num_prompts = len(prompts)

        # process images in batches to avoid exceeding OpenAI API limits
        for start_idx in range(0, total_num_prompts, max_batch_size):
            end_idx = min(start_idx + max_batch_size, total_num_prompts)
            batch_prompts = prompts[start_idx:end_idx]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_batch_size) as executor:
                if llm_model == 'gpt-4':
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_4(prompt, step), batch_prompts))
                else:
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_3p5(prompt, step), batch_prompts))
            responses.extend(batch_responses)

        return responses


    def _query_openai_gpt_3p5(self, prompt, step, verbose=False):
        client = OpenAI()

        if step == 'related_objects':
            messages = self.messages_to_extract_objects_of_interest(prompt)
        else:
            raise ValueError(f'Invalid step: {step}')

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        message = response['choices'][0]['message']['content']
        if verbose:
            print(f'Response: {message}')

        return message
