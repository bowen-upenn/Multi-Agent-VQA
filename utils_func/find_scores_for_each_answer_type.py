import json
import re

# Load ground truth annotations
with open('/tmp/datasets/coco/vqa/v2_mscoco_rest_val2014_annotations.json', 'r') as f:
    ground_truths = json.load(f)

# Initialize counters
correct_counts = {'yes/no': 0, 'number': 0, 'other': 0}
incorrect_counts = {'yes/no': 0, 'number': 0, 'other': 0}


# Function to process each response file
def process_responses(file_path):
    with open(file_path, 'r') as f:
        responses = json.load(f)

    for question_id, response in responses.items():
        if question_id in ground_truths:
            gt = ground_truths[question_id]
            answer_type = gt['answer_type']
            # Assuming the prediction is correct if majority_vote contains [Correct]
            if re.search(r'\[Answer Failed\]', response['initial_answer']) is not None:
                incorrect_counts[answer_type] += 1
            else:
                if re.search(r'\[Correct\]', response['majority_vote']) is not None:
                    correct_counts[answer_type] += 1
                else:
                    incorrect_counts[answer_type] += 1


# List of response files
response_files = [
    # '../outputs/responses1.json',
    '../outputs/responses4_mar18_nocount.json',
    # '../outputs/responses1.json',
    # '../outputs/responses2.json',
    # '../outputs/responses3.json',
    # '../outputs/responses4.json'
]

# Process each response file
for file_path in response_files:
    process_responses(file_path)

total_counts = {k: correct_counts[k] + incorrect_counts[k] for k in correct_counts}
accuracy = {k: correct_counts[k] / total_counts[k] for k in correct_counts}
total_accuracy = sum(correct_counts.values()) / sum(total_counts.values())

# Print results
print("Correct predictions:", correct_counts)
print("Incorrect predictions:", incorrect_counts)
print("Total predictions:", total_counts)
print("Accuracy:", accuracy)
print("Total Accuracy:", total_accuracy)

