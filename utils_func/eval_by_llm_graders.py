from tqdm import tqdm
import sys
import json
import yaml

sys.path.append('../')
from query_llm import QueryLLM
from utils import Grader


def evaluate_llm(outputs):
    try:
        with open('../config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    with open("../openai_key.txt", "r") as api_key_file:
        api_key = api_key_file.read()

    grader = Grader()
    LLM = QueryLLM(args, openai_key=api_key)

    print('Evaluating the model outputs. Total number of outputs: ', len(outputs))
    for idx, output in tqdm(enumerate(outputs)):
        """
        outputs should have the format:
        {
            "question_id": 89273000,
            "answer": "yes",
            "target_answer": "yes",
            "question": "Is this cake homemade?"
        }
        """
        grades = []
        for grader_id in range(3):
            grades.append(LLM.query_llm([output['question']], target_answer=output['target_answer'], model_answer=output['answer'],
                                        step='grade_answer', grader_id=grader_id, verbose=args['inference']['verbose']))

        _ = grader.accumulate_grades_simple(args, grades)

        if (idx + 1) % args['inference']['print_every'] == 0:
            accuracy, _ = grader.average_score_simple()
            print('Accuracy at idx ', idx, ': ', accuracy)


# Load the output file
with open('../unilm/vlmo/result/vqa_submit_vlmo_large_patch16_384_coc_annot.json', 'r') as output_file:
# with open('../unilm/vlmo/result/vqa_submit_vlmo_large_patch16_480_vq_annot.json', 'r') as output_file:
# with open('../unilm/beit3/outputs/submit_gqa_val1000_large_finetuned_annot.json', 'r') as output_file:
    model_outputs = json.load(output_file)

# Evaluate the output file
evaluate_llm(model_outputs)
