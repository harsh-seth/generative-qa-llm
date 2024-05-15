from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MyDataset2 import Dataset

from RACE_Dataset import RaceDataset
from evaluation_metrics import EvaluationMetrics

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

def getTestAccuracy(model, tokenizer, test_set, batch_size=4, workers=0, max_input_length=512, device='cuda', results_file_path="metrics.csv"):
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                                          num_workers=workers, collate_fn=lambda data: test_set.pack_minibatch(data))
    
    model.eval()
    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for contexts, questions, answers in tqdm(test_dataloader):
            inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                questions, contexts)))
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                answers,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            encoded_inputs = encoded_inputs.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            model_predictions = model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1, exact_match = test_set.evaluate(model_predictions_encoded, target_encoded)
    print(f"\t Validation F1 = {f1:.2f}, EM = {exact_match:.2f}")
    with open(results_file_path, 'a') as results_file:
        results_file.write(f"test,{len(model_predictions_encoded)},{f1:.2f},{exact_match:.2f}\n")

    



if __name__ == '__main__':
    device = 'cuda'

    print("Loading model..")
    tokenizer = T5Tokenizer.from_pretrained("results/t5-base/tokenizer/best-f1")
    model = T5ForConditionalGeneration.from_pretrained("results/t5-base/model/best-f1")
    print("Model loaded.")
    model.to(device)
    model.eval()

    print("Loading dataset..")
    raceDataset = RaceDataset()
    test_set = raceDataset.get_dataset('test', num_records=100)
    test_set = Dataset(test_set, tokenizer)
    print("Dataset loaded.")

    question = raceDataset.get_questions('test')[0]
    context = raceDataset.get_context('test')[0]
    correct_answer = raceDataset.get_answer('test')[0]
    
    print("Performing inference for sanity")
    print(f"Question: {question}")
    input_str = f"question: {question}  context: {context}"
    output_str = generateInference(model, tokenizer, input_str)
    print(f"Correct Answer: {correct_answer}")
    print(f"Model Answer: {output_str}")

    print("Getting test split accuracy")
    getTestAccuracy(model, tokenizer, test_set)

    eval_metric = EvaluationMetrics(output_str, correct_answer)
    print("Rouge score: ", eval_metric.get_rouge_score())
    print("Bluert score: ", eval_metric.get_bleurt_score())
    print("BERT score: ")
    eval_metric.get_bert_score()
    print('\nLLM Evaluation: ', eval_metric.LLM_evaluation())
    
