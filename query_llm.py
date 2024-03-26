import torch
import numpy as np
from tqdm import tqdm
import os
import json
import openai
from openai import OpenAI
import random
import cv2
import concurrent.futures
from utils import *
# gemini
from vertexai.preview.generative_models import GenerativeModel

class QueryLLM:
    def __init__(self, args, openai_key=None):
        self.args = args
        self.llm_type = args["model"]
        if openai_key is not None:
            self.api_key = openai_key
        else:
            with open("openai_key.txt", "r") as api_key_file:
                self.api_key = api_key_file.read()


    def message_to_check_if_the_number_is_large(self, answer):
        message = [
            {"role": "system", "content": "Your task is to verify if the number mentioned in the answer after '[Numeric Answer]' is larger than three. "
                                          "If so, say '[Yes]'. Otherwise, say '[No]'."},
            {"role": "user", "content": '[Answer] ' + answer}
        ]
        return message


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


    def messages_to_extract_needed_objects(self, question, previous_response, verify_numeric_answer=False):
        if verify_numeric_answer:
            messages = [
                {"role": "system", "content": "Your task is to identify the object mentioned in the question: '" + question + " that needs to be counted in the image. "
                                              "Please list only those object that needs to be counted, ignoring all other objects mentioned in the question. "
                                              "Format your response as a single line like 'Object1'. This information will assist in directing an additional object counting model to "
                                              "more accurately locate and count the specified object(s). "},
            ]
        else:
            messages = [
                {"role": "system", "content": "Based on the response provided by a large vision-language model (VLM) for a visual question answering task, "
                                              "it appears that the model encountered difficulties in generating an accurate answer for the question: '" + question + "'. "
                                              "The model has provided an explanation for its inability to respond correctly, which might suggest that certain objects "
                                              "important to answer the question were not detected in the image. "
                                              "Your task is to analyze the model's explanation carefully to identify those objects or attributes. "
                                              "For questions asking about specific objects (e.g., 'What is the color of the car?'), list the objects 'Car' directly. "
                                              "For questions seeking objects with certain attributes (e.g., 'Which object has a bright color?'), list the attributes with the word 'objects' (e.g., 'bright-colored objects'). "
                                              "Make sure to include the subject and the object of the question, as they must be critical to answer the question, but "
                                              "ignore objects irrelevant to the question even if they are mentioned in the model explanation. "
                                              "This nuanced approach will guide the deployment of an additional object detection model to locate these missing objects or attributes. "
                                              "If you find no objects from the explanation, you can instead extract the objects mentioned in the question. "
                                              "Always list the objects in the following format in a single line: 'Object1 . Object2 . Object3 .'"},
                {"role": "user", "content": "Here is the explanation from the VLM regarding its failure to answer the question correctly: '" + previous_response + "',"
                                            "and the visual question to be answered is: '" + question + "'."}
            ]
        return messages


    def messages_to_grade_the_answer(self, question, target_answer, model_answer, grader_id=0):
        if grader_id == 0:
            messages = [
                {"role": "system", "content": "Please grade the following answer provided by a large vision-language model (VLM) for a visual question answering task in one to two sentences. "
                              "Please understand that the correct answer provided by the dataset is artificially too short. Therefore, as long as the correct answer is mentioned in the VLM's answer, "
                              "it should be graded as '[Grader 0] [Correct]'. If the VLM's answer contains the correct answer but has additional information not mentioned by the correct answer, it is still '[Correct]'. " 
                              "If the question involves multiple conditions and the correct answer is no, grade the VLM's answer as '[Grader 0] [Correct]' as long as it correctly finds that one of the conditions is not met. "
                              "If the answer is a number, verify if the number is correct. "
                              "Partially correct answer or synonyms is still '[Grader 0] [Correct]'. For example, brown and black are synonyms. Otherwise, if the VLM's answer misses the targeted information, grade the answer as '[Grader 0] [Incorrect]'. "
                              "Focus on the part after '[Answer]' or '[Reattempted Answer]'. Reason your grading step by step but keep it short. "},
                {"role": "user", "content": "The VLM was asked the question: '" + question + "'. "
                                            "The correct answer for the question is: '" + target_answer + "'. "
                                            "The VLM provided the following answer: '" + model_answer + "'. "},
            ]
        elif grader_id == 1:
            messages = [
                {"role": "system", "content": "Evaluate the accuracy of a VLM's response to a visual question. "
                                              "Consider the provided correct answer as a benchmark. If the VLM's response includes the correct answer, even with additional information, rate it as '[Grader 1] [Correct]'. Partially correct answer or synonyms is still '[Grader 1] [Correct]'. For example, brown and black are synonyms."
                                              "For a question that involves multiple criteria, such as 'Does the image contain a brightly colored and large doll?' and the correct answer is 'No', "
                                              "a response like 'The doll indeed has a bright color but it is not large' that correctly identifies at least one criterion not being met, even if other criteria are met, should be rated as '[Correct]'. " 
                                              "A '[Grader 1] [Correct]' rating applies to answers that are partially right. If the VLM fails to address the key point of the question, mark it as '[Grader 1] [Incorrect]'. "
                                              "If the answer is a number, check if the number is correct. "
                                              "Focus on the part after '[Answer]' or '[Reattempted Answer]'. Provide a brief rationale for your evaluation, highlighting the reasoning behind your decision. "},
                {"role": "user", "content": "Question posed to the VLM: '" + question + "'. "
                                            "Dataset's correct answer: '" + target_answer + "'. "
                                            "VLM's response: '" + model_answer + "'. "}
            ]
        else:
            messages = [
                {"role": "system", "content": "Assess the response from a VLM on a visual question task. "
                                              "Use the dataset's correct answer as the standard. A response should be marked '[Grader 2] [Correct]' if it mentions the correct answer, regardless of additional unrelated details. "
                                              "In cases where the question requires satisfying multiple criteria and the answer is negative, the response is '[Correct]' if it correctly finds one criteria that is not met. "
                                              "Even partially accurate responses qualify as '[Grader 2] [Correct]'. Mark the response as '[Grader 2] [Incorrect]' only when it overlooks essential details. "
                                              "If the answer is a number, grade if the number is correct. "
                                              "Focus on the part after '[Answer]' or '[Reattempted Answer]'. Justify your assessment step by step but keep it brief. "},
                {"role": "user", "content": "Visual question asked: '" + question + "'. "
                                            "Correct answer according to the dataset: '" + target_answer + "'. "
                                            "Answer provided by the model: '" + model_answer + "'. "}
            ]
        return messages


    def query_llm(self, prompts, previous_response=None, target_answer=None, model_answer=None, grader_id=0, llm_model='gpt-3.5-turbo', step='related_objects', max_batch_size=4,
                  verify_numeric_answer=False, verbose=False):
        # query on a single image
        if len(prompts) == 1:
            if llm_model == 'gpt-4':
                response = self._query_openai_gpt_4(prompts[0], step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
                                                    grader_id=grader_id, verify_numeric_answer=verify_numeric_answer, verbose=verbose)
            else:
                response = self._query_openai_gpt_3p5(prompts[0], step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
                                                      grader_id=grader_id, verify_numeric_answer=verify_numeric_answer, verbose=verbose)
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
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_4(prompt, step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
                                                                                                grader_id=grader_id, verify_numeric_answer=verify_numeric_answer, verbose=verbose), batch_prompts))
                else:
                    batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt_3p5(prompt, step, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
                                                                                                  grader_id=grader_id, verify_numeric_answer=verify_numeric_answer, verbose=verbose), batch_prompts))
            responses.extend(batch_responses)

        return responses


    def _query_openai_gpt_3p5(self, prompt, step, previous_response=None, target_answer=None, model_answer=None, grader_id=0, verify_numeric_answer=False, verbose=False):
        client = OpenAI(api_key=self.api_key)

        if step == 'check_numeric_answer':
            messages = self.message_to_check_if_the_number_is_large(prompt)
        elif step == 'related_objects':
            messages = self.messages_to_extract_related_objects(prompt)
        elif step == 'needed_objects':
            messages = self.messages_to_extract_needed_objects(prompt, previous_response, verify_numeric_answer)
        elif step == 'grade_answer':
            messages = self.messages_to_grade_the_answer(prompt, target_answer, model_answer, grader_id)
        else:
            raise ValueError(f'Invalid step: {step}')

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )

            response = response.choices[0].message.content
        except:
            response = "Invalid response. "
        if verbose:
            print(f'LLM Response at step {step}: {response}')

        return response
