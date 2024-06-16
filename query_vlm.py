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
import re
# gemini
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image as vertexai_Image

class QueryVLM:
    def __init__(self, args, image_size=512):
        self.image_cache = {}
        self.image_size = image_size
        self.min_bbox_size = args['vlm']['min_bbox_size']
        self.args = args
        self.vlm_type = args["model"]

        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()

        # Init gemini model
        if self.vlm_type=="gemini":
            print("Using Gemini Pro Vision as VLM, initializing the model")
            self.gemini_pro_vision = GenerativeModel("gemini-1.0-pro-vision")

    def process_image(self, image, bbox=None):
        # we have to crop the image before converting it to base64
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if bbox is not None:
            width, height = bbox[2], bbox[3]
            xyxy = box_convert(boxes=bbox, in_fmt="cxcywh", out_fmt="xyxy")
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # increase the receptive field of each box to include possible nearby objects and contexts
            if width < self.min_bbox_size:
                x1 = int(max(0, x1 - (self.min_bbox_size - width) / 2))
                x2 = int(min(image.shape[1], x2 + (self.min_bbox_size - width) / 2))
            if height < self.min_bbox_size:
                y1 = int(max(0, y1 - (self.min_bbox_size - height) / 2))
                y2 = int(min(image.shape[0], y2 + (self.min_bbox_size - height) / 2))

            # cv2.imwrite('test_images/original_image' + str(bbox) + '.jpg', image)
            image = image[y1:y2, x1:x2]
            # cv2.imwrite('test_images/cropped_image' + str(bbox) + '.jpg', image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = np.array(buffer).tobytes()
        # gpt4 need decode to base64, but gemini need raw byte
        if self.vlm_type == "gpt4":
            image_bytes = base64.b64encode(image_bytes).decode('utf-8')

        return image_bytes

    def messages_to_answer_directly_gemini(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            # "list each one by [Object i] where i is the index." \
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + \
                      "', explain what the question wants to ask, what objects or objects with specific attributes is related to the question, " \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, provide your final answer which starts with the notation '[Answer]' at the end. " \
                      "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are four different cases for the final answer part. " \
                      "(Case 1) If you believe your answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Answer]' based on your decision. " \
                      "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                      "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                      "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                      "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                      "Then, step-by-step describe each object in this image that satisfy the descriptions in the question. " \
                      "If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. " \
                      "If there are less or equal to three objects in the image, you should start your final answer with '[Numeric Answer]' instead of '[Answer]', answer your predicted number right after '[Numeric Answer]'. " \
                      "If you believe there are many(larger than three objects) in the image, you should answer '[Non-zero Numeric Answer] [Answer Failed]' instead of '[Answer]', provide your predicted number right after '[Non-zero Numeric Answer]'. Avoid being too confident. " \
                      "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Answer]'. No extra words after '[Answer]'. " \
                      "(Case 4) If you think you can't answer the question directly, or you need more information, or you find that your answer could be wrong, " \
                      "do not make a guess. Instead, explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and answer '[Answer Failed]' instead of '[Answer]'. Keep your answers short."
        else:
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
                      "Begin your final answer with the notation '[Answer]'. " \
                      "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        return message



    def messages_to_answer_directly(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image. " \
                      "Finally, begin your final answer with the notation '[Answer]'. " \
                      "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
                      "(Case 1) If you believe your answer falls into the category of 'yes/no', say 'yes/no' after '[Answer]'. " \
                      "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                      "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                      "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                      "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                      "Begin your answer with '[Numeric Answer]', step-by-step describe each object in this image that satisfy the descriptions in the question, " \
                      "list each one by [Object i] where i is the index, and finally predict the number. " \
                      "If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. " \
                      "If there are many of them, for example, more than three in the image, you should answer '[Non-zero Numeric Answer]' and '[Answer Failed]'. and avoid being too confident. " \
                      "(Case 3) If the answer should be an activity or a noun, say the word after '[Answer]'. Similarly, no extra words after '[Answer]'. " \
                      "(Case 4) If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'. Keep your answers short"
        else:
            message = "You are performing a Visual Question Answering task." \
                      "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
                      "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                      "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
                      "Begin your final answer with the notation '[Answer]'. " \
                      "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                      "do not make a guess, but please explain why and what you need to solve the question," \
                      "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        return message


    def message_to_check_if_answer_is_numeric(self, question):
        message = "You are performing a Visual Question Answering task. " \
                  "Given the image and the question '" + question + "', please first verify if the question type is like 'how many' or 'what number of' and asks you to count the number of an object. " \
                  "If not, say '[Not Numeric Answer]' and explain why. " \
                  "Otherwise, find which object you need to count, say '[Numeric Answer]', and predict the number. "
        return message


    def messages_to_query_object_attributes(self, question, phrase=None, verify_numeric_answer=False):
        if verify_numeric_answer:
            message = "Describe the " + phrase + " in each image in one sentence that can help you answer the question '" + question + "' and count the number of " + phrase + " in the image. "
        else:
            # We expect each object to offer a different perspective to solve the question
            # message = "Describe the attributes and the name of the object related to answer the question '" + question + "' in one sentence."
            message = "Describe the attributes and the name of the object in the image in one sentence, " \
                  "including visual attributes like color, shape, size, materials, and clothes if the object is a person, " \
                  "and semantic attributes like type and current status if applicable. " \
                  "Think about what objects you should look at to answer the question '" + question + "' in this specific image, and only focus on these objects." \

            if phrase is not None:
                message += "You need to focus on the " + phrase + " and nearby objects. "

        return message


    def messages_to_reattempt(self, question, obj_descriptions, prev_answer):
        # message = "After a previous attempt to answer the question '" + question + "', the response was not successful, " \
        #           "Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "]. To address this, we've identified additional objects within the image: "                                                                                                                                                                                                                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
        message = "After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
                  "highlighting the need for more detailed object detection and analysis. Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "] " \
                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "

        for i, obj in enumerate(obj_descriptions):
            message += "[Object " + str(i) + "] " + obj + "; "

        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Summarize all the information you have, and then begin your final answer with '[Reattempted Answer]'." \
                       "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
                       "If you believe your answer falls into the category of 'yes/no' or a number, say 'yes/no' or the number after '[Reattempted Answer]'. " \
                       "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                       "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                       "If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                       "step-by-step describe each object in this image that satisfy the descriptions in the question, list each one by [Object i] where i is the index, " \
                       "and finally reevaluated the number after '[Reattempted Answer]'. Objects could be only partially visible." \
                       "If the answer should be an activity or a noun, say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
        else:
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Begin your final answer with '[Reattempted Answer]'."

        return message


    def messages_to_reattempt_gemini(self, question, obj_descriptions, prev_answer):
        # message = "After a previous attempt to answer the question '" + question + "', the response was not successful, " \
        #           "Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "]. To address this, we've identified additional objects within the image: "                                                                                                                                                                                                                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
        message = "You are performing a Visual Question Answering task. After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
                  "highlighting the need for more detailed object detection and analysis. Here is the feedback from that previous attempt for reference: [Previous Failed Answer: " + prev_answer + "] " \
                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "

        for i, obj in enumerate(obj_descriptions):
            message += "[Object " + str(i) + "] " + obj + "; "

        if self.args['datasets']['dataset'] == 'vqa-v2':
            # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
            # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
            message += "Based on the previous attempt, these descriptions and the image, you need to first list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
                       "Then, given these additional object descriptions that the model previously missed, summarize all the information and re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Finally, provide your own answer to the question: '" + question + "'. Your final answer should starts with notation '[Reattempted Answer]'." \
                       "Your final answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are three different cases for the final answer part. " \
                       "(Case 1) If you believe your final answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Reattempted Answer]'. " \
                       "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                       "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                       "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                       "describe each object in this image that related the descriptions in the question. " \
                       "Finally re-evaluated and say the number after '[Reattempted Answer]'. Objects could be only partially visible." \
                       "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
        else:
            message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
                       "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
                       "Begin your final answer with '[Reattempted Answer]'."

        return message


    def query_vlm(self, image, question, step='attributes', phrases=None, obj_descriptions=None, prev_answer=None, bboxes=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        responses = []

        if step == 'reattempt' or step == 'ask_directly' or bboxes is None or len(bboxes) == 0:
            if self.vlm_type=="gemini":
                response = self._query_gemini_pro_vision(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            else:
                response = self._query_openai_gpt_4v(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer,
                                                     verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            return [response]

        # query on a single object
        if len(bboxes) == 1:
            bbox = bboxes.squeeze(0)
            phrase = phrases[0]
            if self.vlm_type == "gemini":
                response = self._query_gemini_pro_vision(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            else:
                response = self._query_openai_gpt_4v(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            responses.append(response)

        else:
            # process all objects from the same image in a parallel batch
            total_num_objects = len(bboxes)
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
                if self.vlm_type == "gemini":
                    response = list(executor.map(lambda bbox, phrase: self._query_gemini_pro_vision(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                else:
                    response = list(executor.map(lambda bbox, phrase: self._query_openai_gpt_4v(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                responses.append(response)

        return responses


    def _query_openai_gpt_4v(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        # we have to crop the image before converting it to base64
        base64_image = self.process_image(image, bbox)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly(question)
            max_tokens = 400
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question)
            else:
                messages = self.messages_to_query_object_attributes(question, phrase)
            max_tokens = 400
        elif step == 'reattempt':
            messages = self.messages_to_reattempt(question, obj_descriptions, prev_answer)
            max_tokens = 600
        else:
            raise ValueError('Invalid step')

        # Retry if GPT response is like "I'm sorry, I cannot assist with this request"
        for _ in range(3):
            # Form the prompt including the image.
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            # print('Prompt: ', messages)
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
            try:
                response_json = response.json()
            except:
                continue

            # Process the response
            # Check if the response is valid and contains the expected data
            if 'choices' in response_json and len(response_json['choices']) > 0:
                completion_text = response_json['choices'][0].get('message', {}).get('content', '')

                if verbose:
                    print(f'VLM Response at step {step}: {completion_text}')
            else:
                completion_text = ""

            if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
                break

        return completion_text


    def _query_gemini_pro_vision(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        byte_image = self.process_image(image, bbox)
        vertexai_image = vertexai_Image.from_bytes(byte_image)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly_gemini(question)
            max_tokens = 500
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question)
            else:
                messages = self.messages_to_query_object_attributes(question,
                                                                    phrase)
            max_tokens = 400
        elif step == 'reattempt':
            messages = self.messages_to_reattempt_gemini(question, obj_descriptions,
                                                  prev_answer)
            max_tokens = 700
        else:
            raise ValueError('Invalid step')

        # Retry if GPT response is like "I'm sorry, I cannot assist with this request"
        completion_text = ""
        for _ in range(3):
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            gemini_contents = [messages, vertexai_image]
            response = self.gemini_pro_vision.generate_content(
                gemini_contents,
                generation_config={"max_output_tokens": max_tokens},
            )
            # Process the response
            try:
                completion_text = response.text
            except:
                if verbose:
                    print(f'Gemini VLM Response at step {step}')
                continue
            if verbose:
                print(f'Gemini VLM Response at step {step}: {completion_text}')

            if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
                break

        return completion_text