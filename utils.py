import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import re

from groundingdino.util.inference import annotate


class Grader:
    def __init__(self):
        self.count_correct = 0
        self.count_incorrect = 0
        self.count_correct_baseline = 0
        self.count_incorrect_baseline = 0
        self.count_total = 0

    def average_score(self):
        """Calculate and return the average score of the grades."""
        if self.count_total == 0:
            return 0  # Return 0 if there are no grades to avoid division by zero

        accuracy_baseline = self.count_correct_baseline / self.count_total
        accuracy = self.count_correct / self.count_total
        return accuracy_baseline, accuracy


def calculate_iou_batch(a, b):
    """
    Vectorized calculation of IoU for pairs of bounding boxes in a and b.
    Parameters:
    - a: PyTorch tensor of shape (N, 4), representing bounding boxes.
    - b: PyTorch tensor of shape (M, 4), representing bounding boxes.
    Returns:
    - iou: PyTorch tensor of shape (M, N), IoU values.
    """
    # Expand dimensions to support broadcasting: (N, 1, 4) with (1, M, 4)
    a = a.unsqueeze(1)  # Shape: (N, 1, 4)
    b = b.unsqueeze(0)  # Shape: (1, M, 4)
    print('a', a.shape, a, 'b', b.shape, b)

    # Calculate intersection coordinates
    max_xy = torch.min(a[..., 2:], b[..., 2:])
    min_xy = torch.max(a[..., :2], b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[..., 0] * inter[..., 1]

    # Calculate areas
    a_area = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    b_area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    # Calculate union
    union = a_area + b_area - intersection

    # Compute IoU
    iou = intersection / union
    print('iou', iou.shape)

    return iou


def filter_boxes_pytorch(a, b, iou_threshold=0.5):
    """
    Filters boxes in b based on IoU threshold with boxes in a using PyTorch.
    Parameters:
    - a, b: PyTorch tensors of shapes (N, 4) and (M, 4) respectively.
    - iou_threshold: float, threshold for filtering.
    Returns:
    - filtered_b: PyTorch tensor of filtered bounding boxes from b.
    """
    iou = calculate_iou_batch(a, b)  # Shape: (M, N)
    # Check if any IoU value exceeds the threshold for each box in b
    max_iou, _ = torch.max(iou, dim=0)
    keep = max_iou > iou_threshold
    print('b', b.shape, 'keep', keep.shape, keep, 'iou', max_iou)
    return b[keep]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    # ax.imshow(img)
    plt.imsave('test_images/masks.jpg', img)


def plot_grounding_dino_bboxes(image_source, boxes, logits, phrases, filename):
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[:, :, [2, 1, 0]]  # BGR2RGB
    plt.imsave('test_images/bboxes' + filename + '.jpg', annotated_frame)


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def accumulate_grades(args, grader, grades, match_baseline_failed):
    # accumulate the grades
    count_match_correct = 0
    for grade in grades:
        if re.search(r'\[Correct\]', grade):
            count_match_correct += 1
    match_correct = True if count_match_correct >= 2 else False  # majority vote: if at least 2 out of 3 graders agree, the answer is correct
    if args['inference']['verbose']:
        if match_correct:
            majority_vote = 'Majority vote is [Correct] with a score of ' + str(count_match_correct)
            print(f'{Colors.OKBLUE}{majority_vote}{Colors.ENDC}')
        else:
            majority_vote = 'Majority vote is [Incorrect] with a score of ' + str(count_match_correct)
            print(f'{Colors.FAIL}{majority_vote}{Colors.ENDC}')

    grader.count_total += 1
    if not match_baseline_failed:  # if the baseline does not fail
        if match_correct:
            grader.count_correct_baseline += 1
            grader.count_correct += 1  # no need to reattempt the answer
        else:
            grader.count_incorrect_baseline += 1
            grader.count_incorrect += 1  # still didn't reattempt the answer in this case
    else:  # if the baseline fails, reattempt the answer
        grader.count_incorrect_baseline += 1
        if match_correct:
            grader.count_correct += 1
        else:
            grader.count_incorrect += 1


def load_answer_list(answer_list_path):
    # Load the answer list from the JSON file
    file_path = '/tmp/datasets/vqa/answer_list.json'
    with open(file_path, 'r') as file:
        answer_list = json.load(file)
    return answer_list


def filter_response(response, answer_list):
    """
    Filters a response from an LLM to only include words that are in the provided answer list.

    Parameters:
    - response (str): The text response from the LLM.
    - answer_list (list): A list of strings containing acceptable answers.

    Returns:
    - str: A filtered response containing only words from the answer_list.
    """
    # Tokenize the response into words
    response_words = response.split()

    # Filter words based on the answer list
    filtered_words = [word for word in response_words if word in answer_list]

    # Join the filtered words back into a string
    filtered_response = ' '.join(filtered_words)

    return filtered_response



