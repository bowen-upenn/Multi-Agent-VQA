import json
from tqdm import tqdm

# Load the model's output file
with open('../unilm/vlmo/result/vqa_submit_vlmo_large_patch16_384_coc.json', 'r') as model_output_file:
# with open('../unilm/beit3/outputs/submit_vqav2_rest_val_large_itc.json', 'r') as model_output_file:
    model_outputs = json.load(model_output_file)

# Load the ground truth annotations
with open('/tmp/datasets/coco/vqa/v2_mscoco_val2014_annotations.json', 'r') as gt_file:
    ground_truths = json.load(gt_file)['annotations']
print('Example ground_truths format:', ground_truths[0])

with open('/tmp/datasets/coco/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r') as gt_file:
    questions = json.load(gt_file)['questions']
print('Example questions format:', questions[0])

# Update the model's outputs with the ground truth answers
for model_output in tqdm(model_outputs):
    question_id = model_output['question_id']

    found = False
    for gt in ground_truths:
        if gt['question_id'] == question_id:
            model_output['target_answer'] = gt['multiple_choice_answer']
            found = True
            break
    if not found:
        print("question id not found in ground_truths", question_id)

    found = False
    for q in questions:
        if q['question_id'] == question_id:
            model_output['question'] = q['question']
            found = True
            break
    if not found:
        print("question id not found in answers", question_id)

# Write the updated outputs back to the file
with open('../unilm/vlmo/result/vqa_submit_vlmo_large_patch16_384_coc_annot.json', 'w') as updated_output_file:
# with open('../unilm/beit3/outputs/submit_vqav2_rest_val_large_itc_annot.json', 'w') as updated_output_file:
    json.dump(model_outputs, updated_output_file, indent=4)
