from rouge import Rouge
from bert_score import score
from evaluate import load

import google.generativeai as genai


class EvaluationMetrics():

    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual

    def get_rouge_score(self):
        rouge = Rouge()
        scores = rouge.get_scores(self.pred, self. actual)
        return scores

    def get_bleurt_score(self):
        metric = load("bleurt")
        # metric = datasets.load_metric("bleurt")
        results = metric.compute(predictions = [self.pred], references = [self.actual])
        return results["scores"][0]


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
        print("Precision:", p.mean().item())
        print("Recall:", r.mean().item())
        print("F1 score:", f1.mean().item())

    def LLM_evaluation(self):
        file_path = 'API_KEY.txt'
        with open(file_path, 'r') as file:
            api_key = file.read().strip() 

        genai.configure(api_key = api_key)

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Predicted Answer: {self.pred} \n Correct Answer: {self.actual}. \n On a scale of 1-5 rate how similar the predicted answer and correct answer are.")

        return response.text


