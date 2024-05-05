from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForQuestionAnswering, AutoTokenizer, MT5ForConditionalGeneration
import torch

from RACE_Dataset import *

def question_answer(model, tokenizer, question, text, device):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text, max_length=512, truncation=True)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1
    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]).to(device), token_type_ids=torch.tensor([segment_ids]).to(device))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    answer = ""
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    
    return answer

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
        