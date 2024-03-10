import json
from tqdm import tqdm

# Load the model's output file
with open('../unilm/vlmo/result/gqa_submit_vlmo_large_patch16_480_vq.json', 'r') as model_output_file:
# with open('../unilm/beit3/outputs/submit_gqa_val1000_large_finetuned.json', 'r') as model_output_file:
    model_outputs = json.load(model_output_file)

# Load the ground truth annotations
with open('/tmp/datasets/gqa/gqasubset1000.json', 'r') as gt_file:
    ground_truths = json.load(gt_file)
print('Example ground_truths format:', ground_truths[0])

# Update the model's outputs with the ground truth answers
for model_output in tqdm(model_outputs):
    question_id = model_output['question_id']

    found = False
    for gt in ground_truths:
        if gt['question_id'] == question_id:
            model_output['target_answer'] = gt['answer']
            model_output['target_full_answer'] = gt['fullAnswer']
            model_output['question'] = gt['question']
            found = True
            break

    if not found:
        print("question id not found in ground_truths", question_id)

# Write the updated outputs back to the file
# with open('../unilm/beit3/outputs/submit_gqa_val1000_large_finetuned_annot.json', 'w') as updated_output_file:
with open('../unilm/vlmo/result/gqa_submit_vlmo_large_patch16_480_vq_annot.json', 'w') as updated_output_file:
    json.dump(model_outputs, updated_output_file, indent=4)
