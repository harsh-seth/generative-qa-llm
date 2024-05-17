def construct_prompt(question, context, answer="", v1=False):
    if v1:
        return f"question: {question} context: {context}"
    return f"Answer the following question with provided context\n##Question: {question}\n##Context: {context}\n##Answer: {answer}"

def dataset_collator(examples, preprocessed=False):
    prompts = []
    responses = []
    option_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

    for example in examples:
        if preprocessed:
            context, question, response = example
        else:
            option = example["options"]
            answer = example["answer"]
            context = example["article"]
            question = example["question"]
            response = option[option_index[answer]]
        responses.append(response)
        prompts.append(construct_prompt(question, context, ""))

    return prompts, responses
