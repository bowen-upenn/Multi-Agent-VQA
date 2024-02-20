import json
from tqdm import tqdm

"""
Given vqa.val1000.jsonl or vqa.rest_val.jsonl used in BEiT3 model as a validation subset,
the purpose of this script is to find the corresponding questions and annotations in vqa.val1000.jsonl
and ensure that we are evaluating our algorithm on the same subset for a fair comparison.
"""

# Load vqa.val1000.jsonl
vqa_annotations = []
with open('/tmp/datasets/coco/vqa.rest_val.jsonl', 'r') as file:
    for line in file:
        vqa_annotations.append(json.loads(line))

# Load v2_OpenEnded_mscoco_val2014_questions.json
with open('/tmp/datasets/coco/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as file:
    coco_questions = json.load(file)

# Function to extract image_id from the image_path
def extract_image_id(image_path):
    parts = image_path.split('_')[-1]  # Split by underscore and take the last part
    image_id = int(parts.split('.')[0])  # Remove file extension and convert to int
    return image_id

# Match annotations
matched_questions = []
for vqa_ann in tqdm(vqa_annotations):
    image_id = extract_image_id(vqa_ann['image_path'])
    qid = vqa_ann['qid']
    for question in coco_questions['questions']:
        if question['image_id'] == image_id and question['question_id'] == qid:
            matched_questions.append(question)
            break  # Assuming each qid is unique, break after finding a match

# Assuming we have the matched_questions from the previous step
# Let's extract the question_ids from matched_questions
matched_question_ids = [question['question_id'] for question in matched_questions]

# Writing matched questions to a new file
with open('/tmp/datasets/coco/vqa/v2_OpenEnded_mscoco_rest_val2014_questions.json', 'w') as outfile:
    json.dump({"questions": matched_questions}, outfile)

print(f"Stored {len(matched_questions)} matched questions in v2_OpenEnded_mscoco_rest_val2014_questions.json")

# ______________________________________________________________________________________________________________________
# Load v2_mscoco_val2014_annotations.json
with open('/tmp/datasets/coco/vqa/v2_mscoco_val2014_annotations.json', 'r') as file:
    coco_annotations = json.load(file)

# Prepare a dict to hold matched annotations keyed by question_id
matched_annotations = {}

for qid in tqdm(matched_question_ids):
    for ann in coco_annotations['annotations']:
        if ann['question_id'] == qid:
            matched_annotations[str(qid)] = ann
            break  # Assuming each question_id is unique, break after finding a match

# Writing matched annotations (as a dictionary) to a new file
with open('/tmp/datasets/coco/vqa/v2_mscoco_rest_val2014_annotations.json', 'w') as outfile:
    json.dump(matched_annotations, outfile, indent=4)  # Use indent for pretty printing

print(f"Stored {len(matched_annotations)} matched annotations in v2_mscoco_rest_val2014_annotations.json")