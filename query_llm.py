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
        self.llm_type = args["llm"]["llm_model"]
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


    def messages_to_check_counting_problem(self, question):
        messages = [
            {"role": "user",
             "content": "Does the question '" + question + "' require counting the number of objects in the image, such as 'how many objects are there' or 'what is the number of'? "
                        "Only consider non-abstract objects. Say '[Yes]' or '[No]'."},
        ]
        return messages


    def messages_to_extract_needed_objects(self, question, previous_response, verify_numeric_answer=False):
        messages = [
            {"role": "user",
             "content": "Would any object detection help with the question '" + question + "' ? If not, simply answer an empty string like ''. "
                        "If so, extract one to two most important objects mentioned in the question: '" + question + "' "
                        "and the previous response: '" + previous_response + "' needed for answering the question. "
                        "For example, if the question asks 'What is the color of the car?', say 'Car'. 'Which object has a bright color?', say 'bright-colored objects'. "                                                                                                                
                        "Always list the objects in the following format in a single line: 'Object1 . Object2 .' No repeated object names, and only non-abstract nouns."},
            # {"role": "system", "content": "A model encountered difficulties in generating an accurate answer for the question: '" + question + "'. "
            #                               "suggesting that certain objects crucial to answer the question were not detected in the image. "
            #                               "Based on the following previous response: " + previous_response + ", what objects are missing?"
            #                               "For example, "
            #                               "Your task is to analyze the model's explanation carefully to identify those objects or attributes. "
            #                               "For questions asking about specific objects (e.g., 'What is the color of the car?'), say 'Car'. "
            #                               "For questions seeking objects with certain attributes (e.g., 'Which object has a bright color?'), say 'bright-colored objects'. "
            #                               "Make sure to include all objects in the question, but ignore those in the previous response irrelevant to the question. "
            #                               "Always list the objects in the following format in a single line: 'Object1 . Object2 . Object3 .'"},
        ]
        # if verify_numeric_answer:
        #     messages = [
        #         {"role": "system", "content": "Your task is to identify the object mentioned in the question: '" + question + " that needs to be counted in the image. "
        #                                       "Please list only those object that needs to be counted, ignoring all other objects mentioned in the question. "
        #                                       "Format your response as a single line like 'Object1'. This information will assist in directing an additional object counting model to "
        #                                       "more accurately locate and count the specified object(s). "},
        #     ]
        # else:
        # messages = [
        #     {"role": "system", "content": "Based on the response provided by a large vision-language model (VLM) for a visual question answering task, "
        #                                   "it appears that the model encountered difficulties in generating an accurate answer for the question: '" + question + "'. "
        #                                   "The model has provided an explanation for its inability to respond correctly, which might suggest that certain objects "
        #                                   "important to answer the question were not detected in the image. "
        #                                   "Your task is to analyze the model's explanation carefully to identify those objects or attributes. "
        #                                   "For questions asking about specific objects (e.g., 'What is the color of the car?'), list the objects 'Car' directly. "
        #                                   "For questions seeking objects with certain attributes (e.g., 'Which object has a bright color?'), list the attributes with the word 'objects' (e.g., 'bright-colored objects'). "
        #                                   "Make sure to include the subject and the object of the question, as they must be critical to answer the question, but "
        #                                   "ignore objects irrelevant to the question even if they are mentioned in the model explanation. "
        #                                   "This nuanced approach will guide the deployment of an additional object detection model to locate these missing objects or attributes. "
        #                                   "If you find no objects from the explanation, you can instead extract the objects mentioned in the question. "
        #                                   "Always list the objects in the following format in a single line: 'Object1 . Object2 . Object3 .'"},
        #     {"role": "user", "content": "Here is the explanation from the VLM regarding its failure to answer the question correctly: '" + previous_response + "',"
        #                                 "and the visual question to be answered is: '" + question + "'."}
        # ]
        return messages


    def messages_to_summarize_reattempt(self, question, reattempt_answer):
        messages = [
            {"role": "system", "content": "Please summarize this answer using a single word or phrase, in order to answer the question: '" + question + "'."
                                          "Here is the current answer: '" + reattempt_answer + "'."},
        ]
        return messages


    def messages_to_grade_the_answer(self, question, target_answer, model_answer, grader_id=0):
        if grader_id == 0:
            messages = [
                # {"role": "system", "content": "Please grade the following answer for a visual question answering task in one sentence. "
                #                               "Please understand that the correct answer provided by the dataset is artificially short. Therefore, as long as the target answer is correctly mentioned in the model's answer, "
                #                               "it should be graded as '[Grader 0] [Correct]'. Having additional information is fine. "
                #                               "If the question involves multiple conditions and the correct answer is no, grade the VLM's answer as '[Grader 0] [Correct]' as long as it correctly finds that one of the conditions is not met. "},
                {"role": "system", "content": "Please grade the following answer for a visual question answering task in one sentences. "
                              "Please understand that the correct answer provided by the dataset is artificially too short. Therefore, as long as the correct answer is mentioned in the VLM's answer, "
                              "it should be graded as '[Grader 0] [Correct]'. If the VLM's answer contains the correct answer but has additional information not mentioned by the correct answer, it is still '[Correct]'. "
                              "If the question involves multiple conditions and the correct answer is no, grade the VLM's answer as '[Grader 0] [Correct]' as long as it correctly finds that one of the conditions is not met. "
                              "If the answer is a number, verify if the number is correct. "
                              "Partially correct answer or synonyms is still '[Grader 0] [Correct]'. For example, brown and black are synonyms. Otherwise, if the VLM's answer misses the targeted information, grade the answer as '[Grader 0] [Incorrect]'. "
                              "If the target answer is no and the model says the information is inconclusive or hard to determine, it is [Correct]. Different words with similar meanings are [Correct]."}, # "Focus on the part after '[Answer]' or '[Reattempted Answer]'."},
                {"role": "user", "content": "The VLM was asked the question: '" + question + "'. "
                                            "The correct answer for the question is: '" + target_answer + "'. "
                                            "The VLM provided the following answer: '" + model_answer + "'. "},
            ]
        elif grader_id == 1:
            messages = [
                # {"role": "system", "content": "Is the answer provided by the VLM correct for this visual question answering task? Answer '[Grader 1] [Correct]' or '[Grader 1] [Incorrect]'."},
                {"role": "system", "content": "Evaluate the accuracy of a VLM's response to a visual question in a single sentence. "
                                              "Consider the provided correct answer as a benchmark. If the VLM's response includes the correct answer, even with additional information, rate it as '[Grader 1] [Correct]'."
                                              "For a question that involves multiple criteria, such as 'Does the image contain a brightly colored and large doll?' and the correct answer is 'No', "
                                              "a response like 'The doll indeed has a bright color but it is not large' that correctly identifies at least one criterion not being met, even if other criteria are met, should be rated as '[Correct]'. "
                                              "A '[Grader 1] [Correct]' rating applies to answers that are partially right. If the VLM fails to address the key point of the question, mark it as '[Grader 1] [Incorrect]'. "
                                              "If the answer is a number, check if the number is correct. "
                                              "If the target answer is no and the model says the information is inconclusive or hard to determine, it is [Correct]. Different words with similar meanings are [Correct]."},
                                              # "Focus on the part after '[Answer]' or '[Reattempted Answer]'. "},
                {"role": "user", "content": "Question posed to the VLM: '" + question + "'. "
                                            "Dataset's correct answer: '" + target_answer + "'. "
                                            "VLM's response: '" + model_answer + "'. "}
            ]
        else:
            messages = [
                # {"role": "system", "content": "Grade the VLM's answer for a visual question answering task in one sentence. "
                #                               "Please note that the dataset's correct answer is deliberately concise. Thus, if the model's response accurately includes the target answer, it should be marked as '[Grader 2] [Correct].' "
                #                               "Additional information in the response is acceptable. "
                #                               "When a question contains multiple conditions and the correct answer is 'no,' the VLM's response should also be rated as '[Grader 2] [Correct]' provided it accurately identifies that at least one condition is unfulfilled."},
                {"role": "system", "content": "Assess the response from a VLM on a visual question task in one sentence. "
                                              "Use the dataset's correct answer as the standard. A response should be marked '[Grader 2] [Correct]' if it mentions the correct answer, regardless of additional unrelated details. "
                                              "In cases where the question requires satisfying multiple criteria and the answer is negative, the response is '[Correct]' if it correctly finds one criteria that is not met. "
                                              "Even partially accurate responses qualify as '[Grader 2] [Correct]'. Mark the response as '[Grader 2] [Incorrect]' only when it overlooks essential details. "
                                              "If the answer is a number, grade if the number is correct. "
                                              "If the target answer is no and the model says the information is inconclusive or hard to determine, it is [Correct]. Different words with similar meanings are [Correct]."},
                                              # "Focus on the part after '[Answer]' or '[Reattempted Answer]'. "},
                {"role": "user", "content": "Visual question asked: '" + question + "'. "
                                            "Correct answer according to the dataset: '" + target_answer + "'. "
                                            "Answer provided by the model: '" + model_answer + "'. "}
            ]
        return messages


    def query_llm(self, prompts, previous_response=None, target_answer=None, model_answer=None, grader_id=0, llm_model='gpt-4-turbo', step='related_objects', max_batch_size=4,
                  verify_numeric_answer=False, verbose=False):
        # query on a single image
        if len(prompts) == 1:
            response = self._query_openai_gpt(prompts[0], step, llm_model, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
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
                batch_responses = list(executor.map(lambda prompt: self._query_openai_gpt(prompt, step, llm_model, previous_response=previous_response, target_answer=target_answer, model_answer=model_answer,
                                                                                          grader_id=grader_id, verify_numeric_answer=verify_numeric_answer, verbose=verbose), batch_prompts))
            responses.extend(batch_responses)

        return responses


    def _query_openai_gpt(self, prompt, step, llm_model, previous_response=None, target_answer=None, model_answer=None, grader_id=0, verify_numeric_answer=False, verbose=False):
        client = OpenAI(api_key=self.api_key)

        if step == 'check_numeric_answer':
            messages = self.message_to_check_if_the_number_is_large(prompt)
        elif step == 'related_objects':
            messages = self.messages_to_extract_related_objects(prompt)
        elif step == 'check_counting_problem':
            messages = self.messages_to_check_counting_problem(prompt)
        elif step == 'needed_objects':
            messages = self.messages_to_extract_needed_objects(prompt, previous_response, verify_numeric_answer)
        elif step == 'summarize_reattempt':
            messages = self.messages_to_summarize_reattempt(prompt, previous_response)
        elif step == 'grade_answer':
            messages = self.messages_to_grade_the_answer(prompt, target_answer, model_answer, grader_id)
        else:
            raise ValueError(f'Invalid step: {step}')

        try:
            response = client.chat.completions.create(
                model=llm_model,  # 'gpt-3.5-turbo' or 'gpt-4-turbo'
                messages=messages,
                max_tokens=500
            )
            response = response.choices[0].message.content

        except:
            response = "Invalid response. "
        if verbose:
            print(f'LLM Response at step {step}: {response}')

        return response
