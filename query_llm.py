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
    def __init__(self, args):
        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()
            self.args = args


    def messages_to_extract_related_objects(self, question):
        messages = [
            {"role": "system", "content": "Please extract object mentioned in the following sentence. "
                                        "For this task, focus solely on tangible objects that can be visually identified in an image, "
                                        "and do not include names like 'Image' if the question asks if there are any certain objects in the image. "
                                        "Separate each object with '.' if there are more than one objects. "
                                        "For example, in the sentence '[Question] Is there a red apple on the table?' you should extract 'Red apple. Table.' "
                                        "and in the sentence '[Question] Are these animals of the same species?' you should extract 'Animals' "},
            {"role": "user", "content": '[Question] ' + question}
        ]
        return messages


    def messages_to_extract_needed_objects(self, question, previous_response):
        messages = [
            {"role": "system", "content": "Based on the response provided by a large vision-language model (VLM) for a visual question answering task, "
                          "it appears that the model encountered difficulties in generating an accurate answer for the question: '" + question + "'. "
                          "The model has provided an explanation for its inability to respond correctly, which might suggest that certain objects "
                          "important to answer the question were not detected in the image. "
                          "Your task is to analyze the model's explanation carefully to identify and list those objects, "
                          "while ignoring other objects irrelevant to the question even if they are mentioned in the explanation. "
                          "Make sure to include the subject and the object of the question, as they must be critical to answer the question. "
                          "This information will guide the deployment of an additional object detection model to locate these missing objects. "
                          "Here is the explanation from the VLM regarding its failure to answer the question correctly: '" + previous_response + "' "
                          "If you find no objects from the explanation, you can instead extract the objects mentioned in the question. "
                          "List the objects in the following format in a single line: 'Object1 . Object2 . Object3 .'"},
        ]
        return messages


    def messages_to_grade_the_answer(self, question, target_answer, model_answer):
        messages = [
            {"role": "system", "content": "Please grade the following answer provided by a large vision-language model (VLM) for a visual question answering task in one to two sentences. "
                          "The VLM was asked the following question: '" + question + "'. "
                          "The correct answer for the question is: '" + target_answer + "'. "
                          "The VLM provided the following answer: '" + model_answer + "'. "
                          "Please understand that the correct answer provided by the dataset is artificially too short. Therefore, as long as the correct answer is mentioned in the VLM's answer, "
                          "it should be graded as '[Correct]'. If the VLM's answer contains the correct answer but has additional information not mentioned by the correct answer, it is still '[Correct]'. " 
                          "If the question involves multiple conditions and the correct answer is no, grade the VLM's answer as '[Correct]' as long as it correctly finds that one of the conditions is not met. "
                          "Partially correct is still '[Correct]'. Otherwise, if the VLM's answer misses the targeted information, grade the answer as '[Incorrect]'. Reason your grading step by step. "},
        ]
        # "Because the correct answer provided is always too short, please rephrase the question and the correct answer into one declarative sentences. "
        # "For example, if the question asks 'Do the dolls that are to the left of the people look large and colorful?' and the correct answer is 'no', "
        # "you should rephrase the question as 'The dolls that are to the left of the people look are not large and colorful.' "
        return messages


    def query_llm(self, prompts, previous_response=None, target_answer=None, model_answer=None, llm_model='gpt-3.5-turbo', step='related_objects', max_batch_size=4):
        # query on a single image
        if len(prompts) == 1:
            if llm_model == 'gpt-4':
                response = self._query_openai_gpt_4(prompts[0], step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer)
            else:
                response = self._query_openai_gpt_3p5(prompts[0], step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer)
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
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_4(prompt, step, previous_response=previous_response,
                                                                                                target_answer=target_answer, model_answer=model_answer), batch_prompts))
                else:
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_3p5(prompt, step, previous_response=previous_response,
                                                                                                  target_answer=target_answer, model_answer=model_answer), batch_prompts))
            responses.extend(batch_responses)

        return responses


    def _query_openai_gpt_3p5(self, prompt, step, previous_response=None, target_answer=None, model_answer=None, verbose=True):
        client = OpenAI(api_key=self.api_key)

        if step == 'related_objects':
            messages = self.messages_to_extract_related_objects(prompt)
        elif step == 'needed_objects':
            messages = self.messages_to_extract_needed_objects(prompt, previous_response)
        elif step == 'grade_answer':
            messages = self.messages_to_grade_the_answer(prompt, target_answer, model_answer)
        else:
            raise ValueError(f'Invalid step: {step}')

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        message = response.choices[0].message.content
        if verbose:
            print(f'Response: {message}')

        return message
