from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer, MT5ForConditionalGeneration
import torch

from RACE_Dataset import *

if __name__ == '__main__':
    
    device = 'cuda'

    print("Loading model..")
    tokenizer = T5Tokenizer.from_pretrained("results/t5-base/tokenizer/best-f1")
    model = T5ForConditionalGeneration.from_pretrained("results/t5-base/model/best-f1")
    print("Model loaded.")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        context = get_context('test')[0]
        questions: str = [get_questions('test')[0]]
        for question in questions:
            print(f"Question: {question}")
            input_ids = tokenizer(f"question: {question}  context: {context}", return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            output = model.generate(input_ids)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Correct Answer: {get_answer('test')[0]}")
            print(f"Model Answer: {output}")
        