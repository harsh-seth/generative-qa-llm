from datasets import load_dataset
from typing import List

def extract_data(dataset, mode):
    dict_mode=dataset[mode]
    option=dict_mode['options']
    answer=dict_mode['answer']
    article=dict_mode['article']
    question=dict_mode['question']

    return option, answer, article, question

def answer_engineering(answer, options):
    complete_answer=[]
    option_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    for i in range(len(answer)):
        option_number=answer[i]
        complete_answer.append(options[i][option_index[option_number]])
    print(f'Total length of the dataset: {len(complete_answer)}')
    return complete_answer


# Loading dataset
dataset = load_dataset("ehovy/race", 'all', ignore_verifications=True)

# Extracting the dataset specifics
train_options, train_answer, train_article, train_question = extract_data(dataset, 'train')
test_options, test_answer, test_article, test_question = extract_data(dataset, 'test')
validation_options, validation_answer, validation_article, validation_question = extract_data(dataset, 'validation')

# Getting the correct answers
train_correct_answer = answer_engineering(train_answer, train_options)
test_correct_answer = answer_engineering(test_answer, test_options)
validation_correct_answer = answer_engineering(validation_answer, validation_options)


def get_questions(mode):
    if mode=='train':
        return train_question
    elif mode=='val':
        return validation_question
    else:
        return test_question
    

def get_answer(mode):
    if mode=='train':
        return train_correct_answer
    elif mode=='val':
        return validation_correct_answer
    else:
        return test_correct_answer
    

def get_context(mode):
    if mode=='train':
        return train_article
    elif mode=='val':
        return validation_article
    else:
        return test_correct_answer


def construct_dataset(context, question, answer):
    data: List[str] = []
    for i in range(100):
        data.append([context[i], question[i], answer[i]])
    return data


def get_dataset(mode):
    if mode=='train':
        return construct_dataset(train_article, train_question, train_correct_answer)
    elif mode=='val':
        return construct_dataset(validation_article, validation_question, validation_correct_answer)
    else:
        return construct_dataset(test_article, test_question, test_correct_answer)
    
