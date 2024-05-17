import torch

from peft import get_peft_model, PeftConfig
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from utils.evaluation_metrics import EvaluationMetrics

from data.Dataset import construct_prompt
from data.RACE_Dataset import RaceDataset

def generateInference(model, tokenizer, input_str):
    model.eval()
    with torch.no_grad():
        encoded_inputs = tokenizer(input_str, return_tensors="pt")
        input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

if __name__ == '__main__':
    print("Loading model..")
    device = 'cuda'
    base_model_name = "t5-base"
    checkpoint_number = 6
    tokenizer = T5Tokenizer.from_pretrained(f"results/{base_model_name}/tokenizer/checkpoint-{checkpoint_number}")
    model = T5ForConditionalGeneration.from_pretrained(f"results/{base_model_name}/model/checkpoint-{checkpoint_number}")

    # # # for PEFT models
    # model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    # peft_config = PeftConfig.from_pretrained(f"results/{base_model_name}/checkpoint-{checkpoint_number}/model")
    # model = get_peft_model(model, peft_config)
    # tokenizer = T5Tokenizer.from_pretrained(f"results/{base_model_name}/checkpoint-{checkpoint_number}/tokenizer")

    print("Model loaded.")
    model.to(device)
    model.eval()
    print("Model loaded.")

    print("Loading dataset..")
    raceDataset = RaceDataset()
    print("Dataset loaded.")

    print("Performing inference..")
    for idx in range(0, 10):
        print("\n-----\n")
        question = raceDataset.get_questions('test')[idx]
        context = raceDataset.get_context('test')[idx]
        correct_answer = raceDataset.get_answer('test')[idx]
        print(f"Question: {question}")
        input_str = construct_prompt(question, context)
        output_str = generateInference(model, tokenizer, input_str)
        print(f"Correct Answer: {correct_answer}")
        print(f"Model Answer: {output_str}")

        eval_metric = EvaluationMetrics(output_str, correct_answer)
        print("Rouge score: ", eval_metric.get_rouge_score())
        # print("Bluert score: ", eval_metric.get_bleurt_score())
        # print('LLM Evaluation: ', eval_metric.LLM_evaluation())
        print("\n-----\n")
