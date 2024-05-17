from bert_score import score
from collections import Counter
from evaluate import load
import google.generativeai as genai
from rouge import Rouge
from tqdm import tqdm

def f1_score(prediction_tokens, ground_truth_tokens):
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if len(ground_truth) == len(prediction):
        if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
            return 1
    return 0

def get_eval_scores(tokenizer, predictions, gold_answers):
    f1 = exact_match = 0

    for ground_truths, prediction in tqdm(zip(gold_answers, predictions)):
        # Remove pad token
        tokens_to_remove = {
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id
        }
        prediction = list(filter(lambda token: token not in tokens_to_remove, prediction))
        ground_truths = list(filter(lambda token: token not in tokens_to_remove, ground_truths))
        f1 += f1_score(prediction, ground_truths)
        exact_match += exact_match_score(prediction, ground_truths)
    return 100*f1/len(predictions), 100*exact_match/len(predictions)

class EvaluationMetrics():

    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual

    def get_rouge_score(self):
        rouge = Rouge()
        scores = rouge.get_scores(self.pred, self. actual)
        return scores

    # def get_bleurt_score(self):
    #     metric = load("bleurt")
    #     # metric = datasets.load_metric("bleurt")
    #     results = metric.compute(predictions = [self.pred], references = [self.actual])
    #     return results["scores"][0]


    def get_bert_score(self):

        # Step 1: Prepare your reference and candidate sentences
        references = [self.actual]
        candidates = [self.pred]

        # Step 2: Compute the Conditional BERTScore
        # Set the lang parameter according to the language of your sentences, e.g., 'en' for English
        # Set the model type according to your preference, e.g., 'roberta-base', 'bert-base-uncased', etc.
        # Use the option model_type='bert-base-uncased' for BERT and model_type='roberta-base' for RoBERTa
        # Use the option num_layers=None to include all layers
        # Use the option score_type='conditional' to compute Conditional BERTScore
        # Use the option idf=False to disable IDF weighting (if needed)
        # The returned value is a tuple containing (P, R, F1) scores
        p, r, f1 = score(candidates, references, lang='en', model_type='bert-base-uncased', num_layers=None, idf=False)

        # Step 3: Print or use the BERTScore values
        return p.mean().item(), r.mean().item(), f1.mean().item()

    def LLM_evaluation(self):
        file_path = 'API_KEY.txt'
        with open(file_path, 'r') as file:
            api_key = file.read().strip() 

        genai.configure(api_key = api_key)

        model = genai.GenerativeModel('gemini-pro')
        prompt = """0 - Not at all similar\n
        Criteria: The predicted answer and the correct answer have no common elements or relevance to each other. The prediction is entirely incorrect and does not address any aspect of the correct answer.\n
        Example:\n
        Predicted Answer: "Apple"\n
        Correct Answer: "Bicycle"\n
        \n
        1 - Slightly similar\n
        Criteria: The predicted answer and the correct answer have minimal commonality. The prediction shows a very faint hint of understanding but misses the main point or context of the correct answer.\n
        Example:\n
        Predicted Answer: "Car"\n
        Correct Answer: "Bicycle"\n
        (Both are modes of transportation, but different types.)\n
        Example:\n
        Predicted Answer: "You will find bringing a backpack very helpful"\n
        Correct Answer: "You may have the need to free up your hands while street shopping"\n
        (The predicted answer relates to the main point in the correct answer without being specific about it.)\n
        \n
        2 - Somewhat similar\n
        Criteria: The predicted answer and the correct answer share some elements or concepts but differ significantly in details or context. The prediction partially addresses the correct answer but lacks accuracy.\n
        Example:\n
        Predicted Answer: "Motorcycle"\n
        Correct Answer: "Bicycle"\n
        (Both are two-wheeled vehicles, but one is motorized and the other is not.)\n
        Example:\n
        Predicted Answer: "The importance of pies has changed over time"\n
        Correct Answer: "People have paid more and more attention to the importance of pies"\n\n
        (Both talk about the importance of dresses changing, but the correct answer was more specific about what changed.)\n
        \n
        3 - Moderately similar\n
        Criteria: The predicted answer and the correct answer are fairly aligned. The prediction captures the main idea but contains some inaccuracies or missing details. The overall context is mostly understood.\n
        Example:\n
        Predicted Answer: "Mountain Bike"\n
        Correct Answer: "Bicycle"\n
        (A mountain bike is a type of bicycle, indicating a moderate level of similarity.)\n
        \n
        4 - Very similar\n
        Criteria: The predicted answer and the correct answer are closely matched. The prediction is nearly accurate with only minor discrepancies in details. The main idea and most specifics are correctly captured.\n
        Example:\n
        Predicted Answer: "She might be laughed at by the others"\n
        Correct Answer: "She did not want to be laughed at"\n
        (Both answers convey the same meaning.)\n
        Example:\n
        Predicted Answer: "Road Bicycle"\n
        Correct Answer: "Bicycle"\n
        (A road bicycle is a specific type of bicycle, showing a high degree of similarity.)\n
        \n
        5 - Exactly similar\n
        Criteria: The predicted answer and the correct answer are identical or effectively indistinguishable. The prediction is completely accurate and matches the correct answer in all aspects.\n
        Example:\n
        Predicted Answer: "Bicycle"\n
        Correct Answer: "Bicycle"\n
        On a scale of 0-5, evaluate the degree of similarity between the predicted answer and the correct answer. Please give just a number for your score"""
        response = model.generate_content(f"Predicted Answer: {self.pred} \n Correct Answer: {self.actual}. \n {prompt}")

        return response.text


