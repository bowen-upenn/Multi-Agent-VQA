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
from openai import OpenAI
# gemini
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image as vertexai_Image
# llava
import replicate


class QueryVLM:
    def __init__(self, args, image_size=512):
        self.image_cache = {}
        self.image_size = image_size
        self.min_bbox_size = args['vlm']['min_bbox_size']
        self.args = args
        self.vlm_type = args["vlm"]['vlm_model']

        if re.search(r'gemini', self.vlm_type) is not None:
            print("Using Gemini Pro Vision as VLM, initializing the model")
            self.gemini_pro_vision = GenerativeModel("gemini-1.0-pro-vision")
        elif re.search(r'llava', self.vlm_type) is not None:
            with open("replicate_key.txt", "r") as replicate_key_file:
                replicate_key = replicate_key_file.read()
            os.environ['REPLICATE_API_TOKEN'] = replicate_key

        with open("openai_key.txt", "r") as api_key_file:
            self.api_key = api_key_file.read()
            os.environ['OPENAI_API_KEY'] = self.api_key

        with open("replicate_key.txt", "r") as replicate_key_file:
            replicate_key = replicate_key_file.read()
        os.environ['REPLICATE_API_TOKEN'] = replicate_key



    def process_image(self, image_path, bbox=None):
        def _crop_image(bbox, image):
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
            return image


        if re.search(r'llava', self.vlm_type) is not None:
            # Open the image file in binary mode and read it
            with open(image_path, "rb") as file:
                image_data = file.read()
                data = base64.b64encode(image_data).decode('utf-8')
                image_base64 = f"data:application/octet-stream;base64,{data}"

            if bbox is not None:
                # Convert the binary data to cv2
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = _crop_image(bbox, image)

                # Encode the cropped image back to bytes
                _, buffer = cv2.imencode('.jpg', image)
                image_data = base64.b64encode(buffer).decode('utf-8')
                image_base64 = f"data:application/octet-stream;base64,{image_data}"
            return image_base64

        else:
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if bbox is not None:
                image = _crop_image(bbox, image)

            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = np.array(buffer).tobytes()

            # gpt4 need decode to base64, but gemini need raw byte
            if re.search(r'gpt', self.vlm_type) is not None:
                image_bytes = base64.b64encode(image_bytes).decode('utf-8')

            return image_bytes


    def messages_to_answer_directly(self, question):
        if self.args['datasets']['dataset'] == 'vqa-v2':
            additional_instruction = "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number." \
                                     "If the answer should be a number, just say the number. If the answer should be a 'yes' or 'no', just say 'yes' or 'no'. " \
                                     "Note that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'."
                                     # "If the answer should be a number, but you can't find any of such object, you should answer '[Zero Numeric Answer]'. Otherwise, answer '[Non-zero Numeric Answer]'."
        elif self.args['datasets']['dataset'] == 'gqa':
            additional_instruction = ""
        elif self.args['datasets']['dataset'] == 'vqa-synthetic-dataset':
            additional_instruction = ""
        else:
            raise ValueError('Invalid dataset')

        if self.args['inference']['prompt_type'] == 'baseline':
            # no multi-agent pipeline in this mode. prompt reference https://arxiv.org/pdf/2310.03744
            message = question + " Answer the question using a single word or phrase."

        elif self.args['inference']['prompt_type'] == 'baseline_longer':
            # no multi-agent pipeline in this mode. request an answer longer than a single word or phrase
            message = question + " Answer the question using one to two sentences."

        elif self.args['inference']['prompt_type'] == 'simple':  #+ additional_instruction + " " \
            message = question + " If you can answer the question directly, begin your answer with '[Answer]' and answer the question using one to two sentences." \
                      "If you think you can't answer the question directly, do not make a guess." \
                      "Use the notation '[Answer Failed]'. " \
                      "Explain why, and say what are missing and what you need in order to solve the question in one to three sentences. "

        elif self.args['inference']['prompt_type'] == 'cot':
            # add let's think step by step
            message = question + " Begin your answer with '[Answer]'." \
                      "If you think you can't answer the question directly, do not make a guess." \
                      "Use the notation '[Answer Failed]' instead of '[Answer]'. " \
                      "Explain why, and say what are missing and what you need in order to solve the question. " + additional_instruction + " " \
                      "Let's think step by step. Keep your answer short."

        elif self.args['inference']['prompt_type'] == 'ps':
            # plan and solve prompting. prompt reference https://arxiv.org/pdf/2305.04091
            message = question + " Let's first understand the problem and make a plan. " \
                      "Explain what the question wants to ask. Make a plan on which objects to focus, identifying key features of these objects or crucial relationships between them within the given image " \
                      "in order to answer the question. Begin your answer with '[Answer]'." \
                      "If you think you can't answer the question directly, do not make a guess." \
                      "Use the notation '[Answer Failed]' instead of '[Answer]'. " \
                      "Explain why, and say what are missing and what you need in order to solve the question. " + additional_instruction + " " \
                      "Let's think step by step. Keep your answer short."

        elif self.args['inference']['prompt_type'] == 'detailed':
            if self.vlm_type == "gpt4":
                if self.args['datasets']['dataset'] == 'vqa-v2':
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
                              "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'. Keep your answers short."

            elif self.vlm_type == 'gemini':
                if self.args['datasets']['dataset'] == 'vqa-v2':
                    message = "You are performing a Visual Question Answering task." \
                              "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes is related to the question, " \
                              "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                              "Then, your task is to answer the visual question step by step, provide your final answer which starts with the notation '[Answer]' at the end. " \
                              "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are four different cases for the final answer part. " \
                              "(Case 1) If you believe your answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Answer]' based on your decision. " \
                              "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                              "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                              "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                              "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                              "Then, step-by-step describe each object in this image that satisfy the descriptions in the question. " \
                              "If you can't find any of such object, you should answer '[Zero Numeric Answer]'. " \
                              "If there are less or equal to three objects in the image, you should start your final answer with '[Numeric Answer]' instead of '[Answer]', answer your predicted number right after '[Numeric Answer]'. " \
                              "If you believe there are many(larger than three objects) in the image, you should answer '[Non-zero Numeric Answer] [Answer Failed]' instead of '[Answer]', provide your predicted number right after '[Non-zero Numeric Answer]'. Avoid being too confident. " \
                              "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Answer]'. No extra words after '[Answer]'. " \
                              "(Case 4) If you think you can't answer the question directly, or you need more information, or you find that your answer could be wrong, " \
                              "do not make a guess. Instead, explain why and what you need to solve the question," \
                              "like which objects are missing or you need to identify, and answer '[Answer Failed]' instead of '[Answer]'. Keep your answers short."
        else:
            raise ValueError('Invalid prompt type')
            # else:
            #     if self.vlm_type == "gpt4":
            #         if self.args['inference']['prompt_type'] == 'ps':
            #             # plan and solve prompting. prompt reference https://arxiv.org/pdf/2305.04091
            #             message = question + " Let's first understand the problem and think step by step. " \
            #                       "Explain what the question wants to ask. Make a plan on which objects to focus, identifying key features of these objects or crucial relationships between them within the given image " \
            #                       "in order to answer the question. Begin your final answer with '[Answer]'." \
            #                       "If you think you can't answer the question directly, do not make a guess." \
            #                       "Use the notation '[Answer Failed]' instead of '[Answer]'. " \
            #                       "Explain why, and say what are missing and what you need in order to solve the question. " \
            #                       "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. " \
            #                       "If the answer should be a number, but you can't find any of such object, you should answer '[Zero Numeric Answer]'. Otherwise, answer '[Non-zero Numeric Answer]'. " \
            #                       "Keep your answer short."
                    # else:
                    #     # detailed instructions on how to make plans, how to answer different types of questions, and what to take care of
                    #     message = "You are performing a Visual Question Answering task." \
                    #               "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes" \
                    #               "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                    #               "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image. " \
                    #               "Finally, begin your final answer with the notation '[Answer]'. " \
                    #               "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
                    #               "(Case 1) If you believe your answer falls into the category of 'yes/no', say 'yes/no' after '[Answer]'. " \
                    #               "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                    #               "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                    #               "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                    #               "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                    #               "Begin your answer with '[Numeric Answer]', step-by-step describe each object in this image that satisfy the descriptions in the question, " \
                    #               "list each one by [Object i] where i is the index, and finally predict the number. " \
                    #               "If you can't find any of such object, you should answer '[Zero Numeric Answer]' and '[Answer Failed]'. " \
                    #               "If there are many of them, for example, more than three in the image, you should answer '[Non-zero Numeric Answer]' and '[Answer Failed]'. and avoid being too confident. " \
                    #               "(Case 3) If the answer should be an activity or a noun, say the word after '[Answer]'. Similarly, no extra words after '[Answer]'. " \
                    #               "(Case 4) If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
                    #               "do not make a guess, but please explain why and what you need to solve the question," \
                    #               "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'. Keep your answers short"

                # elif self.vlm_type == "gemini":
                #     if self.args['inference']['prompt_type'] == 'simple_ps':
                #         # plan and solve prompting
                #         message = "You are performing a Visual Question Answering task." \
                #                   "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes is related to the question, " \
                #                   "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                #                   "Then, your task is to answer the visual question step by step, provide your final answer which starts with the notation '[Answer]' at the end. " \
                #                   "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. " \
                #                   "If you think you can't answer the question directly, do not make a guess. Instead, explain why and what you need to solve the question," \
                #                   "like which objects are missing or you need to identify, and answer '[Answer Failed]' instead of '[Answer]'." \
                #                   "If the answer should be a number, but you can't find any of such object, you should answer '[Zero Numeric Answer]', and answer '[Non-zero Numeric Answer]' otherwise. " \
                #                   "Keep your answer short. "
                #     else:
                #         message = "You are performing a Visual Question Answering task." \
                #                   "Given the image and the question '" + question + "', explain what the question wants to ask, what objects or objects with specific attributes is related to the question, " \
                #                   "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
                #                   "Then, your task is to answer the visual question step by step, provide your final answer which starts with the notation '[Answer]' at the end. " \
                #                   "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are four different cases for the final answer part. " \
                #                   "(Case 1) If you believe your answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Answer]' based on your decision. " \
                #                   "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
                #                   "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
                #                   "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
                #                   "pay attention to whether the question has specified any attributes that only a subset of these objects may satisfy, and objects could be only partially visible." \
                #                   "Then, step-by-step describe each object in this image that satisfy the descriptions in the question. " \
                #                   "If you can't find any of such object, you should answer '[Zero Numeric Answer]'. " \
                #                   "If there are less or equal to three objects in the image, you should start your final answer with '[Numeric Answer]' instead of '[Answer]', answer your predicted number right after '[Numeric Answer]'. " \
                #                   "If you believe there are many(larger than three objects) in the image, you should answer '[Non-zero Numeric Answer] [Answer Failed]' instead of '[Answer]', provide your predicted number right after '[Non-zero Numeric Answer]'. Avoid being too confident. " \
                #                   "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Answer]'. No extra words after '[Answer]'. " \
                #                   "(Case 4) If you think you can't answer the question directly, or you need more information, or you find that your answer could be wrong, " \
                #                   "do not make a guess. Instead, explain why and what you need to solve the question," \
                #                   "like which objects are missing or you need to identify, and answer '[Answer Failed]' instead of '[Answer]'. Keep your answers short."
                # else:
                #     raise ValueError('Invalid VLM')

        # elif self.args['datasets']['dataset'] == 'gqa':
        #     # GQA dataset has no object counting questions
        #     if self.args['inference']['prompt_type'] == 'no_instruct':
        #         message = question + " Keep your answers short. "
        #
        #     elif self.args['inference']['prompt_type'] == 'simple':
        #         message = "You are performing a Visual Question Answering task." \
        #                   "Given the image, answer the question '" + question + "'. Begin your answer with '[Answer]'." \
        #                   "If you think you can't answer the question directly, do not make a guess" \
        #                   "Say what you need to solve the question and use the notation '[Answer Failed]' instead of '[Answer]'. " \
        #                   "Keep your answer short. "
        #
        #     elif self.args['inference']['prompt_type'] == 'simple_cot':
        #         # add let's think step by step
        #         message = "You are performing a Visual Question Answering task." \
        #                   "Given the image, answer the question '" + question + "'. Let's think step by step. " \
        #                   "Begin your final answer with '[Answer]'." \
        #                   "If you think you can't answer the question directly, do not make a guess" \
        #                   "Say what you need to solve the question and use the notation '[Answer Failed]' instead of '[Answer]'. " \
        #                   "Keep your answer short. "
        #     else:
        #         # no need for more detailed prompts for GQA dataset
        #         if self.vlm_type == "gpt4":
        #             message = "You are performing a Visual Question Answering task." \
        #                       "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
        #                       "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
        #                       "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
        #                       "Begin your final answer with the notation '[Answer]'. " \
        #                       "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
        #                       "do not make a guess, but please explain why and what you need to solve the question," \
        #                       "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        #
        #         elif self.vlm_type == "gemini":
        #             message = "You are performing a Visual Question Answering task." \
        #                       "Given the image and the question '" + question + "', please first explain what the question wants to ask, what objects or objects with specific attributes" \
        #                       "you need to look at in the given image to answer the question, and what relations between objects are crucial for answering the question. " \
        #                       "Then, your task is to answer the visual question step by step, and verify whether your answer is consistent with or against to the image." \
        #                       "Begin your final answer with the notation '[Answer]'. " \
        #                       "If you think you can't answer the question directly or you need more information, or you find that your answer does not pass your own verification and could be wrong, " \
        #                       "do not make a guess, but please explain why and what you need to solve the question," \
        #                       "like which objects are missing or you need to identify, and use the notation '[Answer Failed]' instead of '[Answer]'."
        #         else:
        #             raise ValueError('Invalid VLM')

        # ##############################################################
        # ## TODO: Add prompts for CLEVR dataset
        # elif self.args['datasets']['dataset'] == 'clevr':
        #     if self.args['inference']['prompt_type'] == 'no_instruct':
        #         message = question + " Keep your answers short. "
        #
        #     elif self.args['inference']['prompt_type'] == 'simple':
        #         message = ""
        #
        #     elif self.args['inference']['prompt_type'] == 'simple_cot':
        #         # add let's think step by step
        #         message = ""
        #     else:
        #         message = ""
        #
        # ## TODO: Add prompts for A-OKVQA dataset
        # elif self.args['datasets']['dataset'] == 'a-okvqa':
        #     if self.args['inference']['prompt_type'] == 'no_instruct':
        #         message = question + " Keep your answers short. "
        #
        #     elif self.args['inference']['prompt_type'] == 'simple':
        #         message = ""
        #
        #     elif self.args['inference']['prompt_type'] == 'simple_cot':
        #         # add let's think step by step
        #         message = ""
        #     else:
        #         message = ""
        # ##############################################################
        # else:
        #     raise ValueError('Invalid dataset')

        return message


    def messages_to_recheck_confidence(self, question, prev_answer):
        message = "Given the following answer from a different user to the question '" + question + "', critic and verify it with the image and " \
                  "say how much you think that this response is correct: '" + prev_answer + "'. " \
                  "Let us think step by step, show your reasoning, and then answer '[Absolutely Correct]', '[Partially Correct]', '[Not Correct]'. " \
                  "Only mention the single option you choose and try to be critical. " \
                  "If the question asks about an object and the previous answer says it can't find any of such objects in the image, NEVER say '[Absolutely Correct]'."
        # message = "Do you need additional help with your previous answer? You must say '[Yes]' or '[No]'. Do not be over-confident or make any guess. " \
        #           "If you couldn't answer the question or you think the previous answer is wrong, you should say '[Yes]', " \
        #           "and list the object names you think that are missing from the image in order to answer the question."
        return message


    def message_to_check_if_answer_is_numeric(self, question):
        message = "Given the image and the question '" + question + "', please first verify if the question type is like 'how many' or 'what number of' and asks you to count the number of an object. " \
                  "If not, say '[Not Numeric Answer]' and explain why. " \
                  "Otherwise, find which object you need to count, say '[Numeric Answer]', and predict the number. "
        return message


    def messages_to_query_object_attributes(self, question, prev_answer, phrase=None, verify_numeric_answer=False):
        if verify_numeric_answer:
            message = "Describe the " + phrase + " in each image in one sentence that can help you answer the question '" + question + "' and count the number of " + phrase + " in the image. "
        else:
            # We expect each object to offer a different perspective to solve the question
            # message = "Describe the attributes and the name of the object related to answer the question '" + question + "' in one sentence."
            message = "Describe the attributes and the name of the object in the image in one sentence. " \
                      "Focus on affirming or criticizing the statement '" + prev_answer + "' with the image or saying it is just unrelated to the question '" + question + "'. " \
                      "Also talks about its relation with objects in its surrounding backgrounds. "
                      # "Only mention attributes related to answering the question '" + question + "' , " \
                      # "such as visual attributes like color, shape, size, materials, and clothes if the object is a person, " \
                      # "and semantic attributes like type and current status if applicable. " \
                      # "Focus on affirm or critic the statement '" + prev_answer + "' with the image or saying it is just unrelated. "
            if phrase is not None:
                message += "Focus on the " + phrase + " and nearby objects. "

        return message


    def messages_to_reattempt(self, question, obj_descriptions, prev_answer, verify_answer):
        message = "Based on the original question '" + question + "', you initially answered '" + prev_answer + "', " \
                  "Another user critics and verifies your answer saying that '" + verify_answer + "'.\n"

        if obj_descriptions is not None:
            message += "The object detection model has now specifically identified related objects in the image and their attributes:\n"

            for i, obj in enumerate(obj_descriptions):
                if re.search(r'sorry', obj.lower()) is None:
                    message += "[Object " + str(i) + "] " + obj + "; "

            message += "Make plans step-by-step on how these information can enhance your initial answer. Analyze relationships between detected objects that are helpful for your answering. " \
                       "Does this additional detailed description provide contradictory information to your previous answer? " \
                       "If yes, please clearly list contradictory information, and verify it with the complete input image to achieve a consensus. " \
                       "Remember that each object is detected independently, so please rely more on the whole image rather than object descriptions when there is a conflict, " \
                       "and trust more on another user's critic. When needed, revise your answer in one to two sentences in the end. " \
                       "If there is no contradictory information, keep your initial answer. Do not create new information here. "

            if self.args['datasets']['dataset'] == 'vqa-v2':
                message += "Remember that the correct answer to the original question could be an open-ended response, a binary decision between 'yes' and 'no', or a number. " \
                           "If the answer should be either 'yes' or 'no' and you are uncertain " \
                           "or the information is inconclusive, insufficient, or indefinite, you should answer 'no'.\n"

            message += "Finally, answer: '" + question + "' using a single word or phrase."
        else:
            message += "Revise your answer in one to two sentences based on another user's critic. Do not create new information here. "

            if self.args['datasets']['dataset'] == 'vqa-v2':
                message += "Remember that the correct answer to the original question could be an open-ended response, a binary decision between 'yes' and 'no', or a number. " \
                           "If the answer should be either 'yes' or 'no' and you are uncertain " \
                           "or the information is inconclusive, insufficient, or indefinite, you should answer 'no'.\n"

            message += "Here is the question again: '" + question + "' Answer a single word or phrase."



        # message += "Does this additional detailed description alter your previous answer? " \
        #            "If no, keep your initial answer. If yes, please revise your answer in one to two sentences. " \
        #            "Remember that the correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. " \
        #            "If the answer should be either 'yes' or 'no' and you are uncertain or the information is inconclusive, you should answer 'no'. " \
        #            "Always begin with '[Reattempted Answer]'"


        # message = "After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
        #           "highlighting the need for more detailed object detection and analysis. Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "] " \
        #           "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
        #
        # for i, obj in enumerate(obj_descriptions):
        #     if re.search(r'sorry', obj.lower()) is None:
        #         message += "[Object " + str(i) + "] " + obj + "; "
        #
        # if self.args['datasets']['dataset'] == 'vqa-v2':
        #     additional_instruction = "The correct answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number." \
        #                              "If the answer should be a number, just say the number. If the answer should be a 'yes' or 'no', just say 'yes' or 'no'. " \
        #                              "Note that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
        #                              "If you still can't decide, say 'no'."
        # elif self.args['datasets']['dataset'] == 'gqa':
        #     additional_instruction = ""
        # else:
        #     raise ValueError('Invalid dataset')
        #
        # message += "Given these additional descriptions that are previously missed, please re-attempt the question '" + question + "" \
        #            "Begin your final answer with '[Reattempted Answer]'. " + additional_instruction + ". Keep your answer short."

        # if self.args['datasets']['dataset'] == 'vqa-v2':
        #     # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
        #     # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
        #     message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
        #                "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
        #                "Summarize all the information you have, and then begin your final answer with '[Reattempted Answer]'." \
        #                "The correct answer could be a 'yes/no', a number, or other open-ended response. " \
        #                "If you believe your answer falls into the category of 'yes/no' or a number, say 'yes/no' or the number after '[Reattempted Answer]'. " \
        #                "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
        #                "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
        #                "If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
        #                "step-by-step describe each object in this image that satisfy the descriptions in the question, list each one by [Object i] where i is the index, " \
        #                "and finally reevaluated the number after '[Reattempted Answer]'. Objects could be only partially visible." \
        #                "If the answer should be an activity or a noun, say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
        # else:
        #     message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
        #                "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
        #                "Begin your final answer with '[Reattempted Answer]'."

        return message


    # def messages_to_reattempt_gemini(self, question, obj_descriptions, prev_answer):
    #     # message = "After a previous attempt to answer the question '" + question + "', the response was not successful, " \
    #     #           "Here is the feedback from that attempt [Previous Failed Answer: " + prev_answer + "]. To address this, we've identified additional objects within the image: "                                                                                                                                                                                                                  "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
    #     message = "You are performing a Visual Question Answering task. After a previous attempt to answer the question '" + question + "' given the image, the response was not successful, " \
    #               "highlighting the need for more detailed object detection and analysis. Here is the feedback from that previous attempt for reference: [Previous Failed Answer: " + prev_answer + "] " \
    #               "To address this, we've identified additional objects within the image. Their descriptions are as follows: "
    #
    #     for i, obj in enumerate(obj_descriptions):
    #         message += "[Object " + str(i) + "] " + obj + "; "
    #
    #     if self.args['datasets']['dataset'] == 'vqa-v2':
    #         # Answers could be 'yes/no', a number, or other open-ended answers in VQA-v2 dataset
    #         # message += "Now, please reattempt to answer the visual question '" + question + "'. Begin your answer with '[Reattempted Answer]'. "
    #         message += "Based on the previous attempt, these descriptions and the image, you need to first list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. " \
    #                    "Then, given these additional object descriptions that the model previously missed, summarize all the information and re-attempt to answer the visual question '" + question + "' step by step. " \
    #                    "Finally, provide your own answer to the question: '" + question + "'. Your final answer should starts with notation '[Reattempted Answer]'." \
    #                    "Your final answer could be an open-ended response, a binary decision between 'yes' and 'no', or a number. So, there are three different cases for the final answer part. " \
    #                    "(Case 1) If you believe your final answer is not an open-ended response, not a number, and should fall into the category of a binary decision between 'yes' and 'no', say 'yes' or 'no' after '[Reattempted Answer]'. " \
    #                    "Understand that the question may not be capture all nuances, so if your answer partially aligns with the question's premises, it is a 'yes'." \
    #                    "For example, if the image shows a cat with many black areas and you're asked whether the cat is black, you should answer 'yes'. " \
    #                    "(Case 2) If the question asks you to count the number of an object, such as 'how many' or 'what number of', " \
    #                    "describe each object in this image that related the descriptions in the question. " \
    #                    "Finally re-evaluated and say the number after '[Reattempted Answer]'. Objects could be only partially visible." \
    #                    "(Case 3) If you believe your answer is an open-ended response(an activity, a noun or an adjective), say the word after '[Reattempted Answer]'. No extra words after '[Reattempted Answer]'"
    #     else:
    #         message += "Based on these descriptions and the image, list any geometric, possessive, or semantic relations among the objects above that are crucial for answering the question and ignore the others. "  \
    #                    "Given these additional object descriptions that the model previously missed, please re-attempt to answer the visual question '" + question + "' step by step. " \
    #                    "Begin your final answer with '[Reattempted Answer]'."
    #
    #     return message


    def query_vlm(self, image, question, step='attributes', phrases=None, obj_descriptions=None, prev_answer=None, verify_answer=None, bboxes=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        responses = []

        if step == 'reattempt' or step == 'ask_directly' or step == 'attributes' or bboxes is None or len(bboxes) == 0:
            if re.search(r'gemini', self.vlm_type) is not None:
                response = self._query_gemini(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer, verify_answer=verify_answer,
                                              verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            elif re.search(r'gpt', self.vlm_type) is not None:
                response = self._query_openai_gpt(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer, verify_answer=verify_answer,
                                                  verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            elif re.search(r'llava', self.vlm_type) is not None or step == 'attributes':
                response = self._query_llava_api(image, question, step, obj_descriptions=obj_descriptions, prev_answer=prev_answer, verify_answer=verify_answer,
                                                 verify_numeric_answer=verify_numeric_answer, needed_objects=needed_objects, verbose=verbose)
            else:
                raise ValueError('Invalid VLM')
            return [response]

        # query on a single object
        if len(bboxes) == 1:
            bbox = bboxes.squeeze(0)
            phrase = phrases[0]
            if re.search(r'gemini', self.vlm_type) is not None:
                response = self._query_gemini(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            elif re.search(r'gpt', self.vlm_type) is not None:
                response = self._query_openai_gpt(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            elif re.search(r'llava', self.vlm_type) is not None:
                response = self._query_llava_api(image, question, step, phrase=phrase, bbox=bbox, verbose=verbose)
            else:
                raise ValueError('Invalid VLM')
            responses.append(response)

        else:
            # process all objects from the same image in a parallel batch
            total_num_objects = len(bboxes)
            with concurrent.futures.ThreadPoolExecutor(max_workers=total_num_objects) as executor:
                if re.search(r'gemini', self.vlm_type) is not None:
                    response = list(executor.map(lambda bbox, phrase: self._query_gemini(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                elif re.search(r'gpt', self.vlm_type) is not None:
                    response = list(executor.map(lambda bbox, phrase: self._query_openai_gpt(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                                needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                elif re.search(r'llava', self.vlm_type) is not None:
                    response = list(executor.map(lambda bbox, phrase: self._query_llava_api(image, question, step, phrase=phrase, bbox=bbox, verify_numeric_answer=verify_numeric_answer,
                                                                                               needed_objects=needed_objects, verbose=verbose), bboxes, phrases))
                else:
                    raise ValueError('Invalid VLM')
                responses.append(response)

        return responses


    def _query_openai_gpt(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        # we have to crop the image before converting it to base64
        # client = OpenAI()
        global messages_ask_directly, response_ask_directly
        base64_image = self.process_image(image, bbox)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly(question)
            messages_ask_directly = messages
            max_tokens = 200
        elif step == 'recheck_confidence':
            messages = self.messages_to_recheck_confidence(question, prev_answer)
            max_tokens = 500
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 200
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question, prev_answer)
            else:
                messages = self.messages_to_query_object_attributes(question, prev_answer, phrase)
            max_tokens = 300
        elif step == 'reattempt':
            messages = self.messages_to_reattempt(question, obj_descriptions, prev_answer, verify_answer)
            max_tokens = 500
        else:
            raise ValueError('Invalid step')
        if len(messages) == 0 or messages is None:
            messages = " "

        for _ in range(3):
            # Form the prompt including the image.
            # Due to the strong performance of the vision model, we omit multiple queries and majority vote to reduce costs
            # print('Prompt: ', messages)
            # if step == 'recheck_confidence':
            #     prompt = {
            #         "model": self.vlm_type,
            #         "messages": [
            #             {
            #                 "role": "user",
            #                 "content": [
            #                     {"type": "text", "text": messages_ask_directly},
            #                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            #                 ]
            #             },
            #             {
            #                 "role": "assistant",
            #                 "content": [
            #                     {"type": "text", "text": response_ask_directly},
            #                 ]
            #             },
            #             {
            #                 "role": "user",
            #                 "content": [
            #                     {"type": "text", "text": messages},
            #                 ]
            #             }
            #         ],
            #         "max_tokens": max_tokens
            #     }
            # else:
            prompt = {
                "model": self.vlm_type,
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
                if step == 'ask_directly':
                    response_ask_directly = completion_text
                return completion_text

            # if step == 'ask_directly' or (not re.search(r'sorry|cannot assist|can not assist|can\'t assist', completion_text, re.IGNORECASE)):
            #     break
        return ""


    def _query_gemini(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        byte_image = self.process_image(image, bbox)
        vertexai_image = vertexai_Image.from_bytes(byte_image)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly(question)
            max_tokens = 500
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            max_tokens = 300
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question, prev_answer)
            else:
                messages = self.messages_to_query_object_attributes(question, prev_answer, phrase)
            max_tokens = 400
        elif step == 'reattempt':
            messages = self.messages_to_reattempt(question, obj_descriptions, prev_answer, verify_answer)
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


    def _query_llava_api(self, image, question, step, phrase=None, bbox=None, obj_descriptions=None, prev_answer=None, verify_numeric_answer=False, needed_objects=None, verbose=False):
        # we have to crop the image before converting it to base64
        base64_image = self.process_image(image, bbox)

        if step == 'ask_directly':
            messages = self.messages_to_answer_directly(question)
            # max_tokens = 200
        elif step == 'check_numeric_answer':
            messages = self.message_to_check_if_answer_is_numeric(question)
            # max_tokens = 200
        elif step == 'attributes':
            if phrase is None or bbox is None:
                messages = self.messages_to_query_object_attributes(question, prev_answer)
            else:
                messages = self.messages_to_query_object_attributes(question, prev_answer, phrase)
            # max_tokens = 300
        elif step == 'reattempt':
            messages = self.messages_to_reattempt(question, obj_descriptions, prev_answer)
            # max_tokens = 300
        else:
            raise ValueError('Invalid step')

        input = {
            "image": base64_image,
            "prompt": messages
        }

        output = replicate.run(
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            input = input
        )
        output = "".join(output)

        if verbose:
            print(f'Gemini VLM Response at step {step}: {output}')

        return output
