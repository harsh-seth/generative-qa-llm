from datasets import load_dataset
from typing import List
import re

class RaceDataset():
    # Loading dataset
    def __init__(self):
        self.dataset = load_dataset("ehovy/race", 'all', ignore_verifications=True)

        # Extracting the dataset specifics
        train_options, train_answer, train_article, train_question = self.extract_data('train')
        test_options, test_answer, test_article, test_question = self.extract_data('test')
        validation_options, validation_answer, validation_article, validation_question = self.extract_data('validation')

        self.train_answer = train_answer
        self.train_article = train_article
        self.train_question = train_question

        self.test_answer = test_answer
        self.test_article = test_article
        self.test_question = test_question

        self.validation_answer = validation_answer
        self.validation_article = validation_article
        self.validation_question = validation_question

        # Getting the correct answers
        self.train_correct_answer = self.answer_engineering(train_answer, train_options)
        self.test_correct_answer = self.answer_engineering(test_answer, test_options)
        self.validation_correct_answer = self.answer_engineering(validation_answer, validation_options)

    def extract_data(self, mode, filter_no_context_qns=False):
        dict_mode=self.dataset[mode]
        options=dict_mode['options']
        answers=dict_mode['answer']
        articles=dict_mode['article']
        questions=dict_mode['question']

        # removing questions which depend on the options
        regex = r"^which of the following|are correct except" # only 13 matches in train set
        for idx, question in enumerate(questions):
            if re.match(regex, question):
                options.pop(idx)
                answers.pop(idx)
                articles.pop(idx)
                questions.pop(idx)

        return options, answers, articles, questions

    def answer_engineering(self, answer, options):
        complete_answer=[]
        option_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        for i in range(len(answer)):
            option_number=answer[i]
            complete_answer.append(options[i][option_index[option_number]])
        print(f'Total length of the dataset: {len(complete_answer)}')
        return complete_answer

    def get_questions(self, mode):
        if mode=='train':
            return self.train_question
        elif mode=='val':
            return self.validation_question
        else:
            return self.test_question
        

    def get_answer(self, mode):
        if mode=='train':
            return self.train_correct_answer
        elif mode=='val':
            return self.validation_correct_answer
        else:
            return self.test_correct_answer
        

    def get_context(self, mode):
        if mode=='train':
            return self.train_article
        elif mode=='val':
            return self.validation_article
        else:
            return self.test_article


    def construct_dataset(self, contexts, questions, answers, num_records=None):
        data: List[str] = []
        for i in range(num_records if num_records else len(questions)):
            data.append([contexts[i], questions[i], answers[i]])
        return data


    def get_dataset(self, mode, num_records=None):
        if mode=='train':
            return self.construct_dataset(self.train_article, self.train_question, self.train_correct_answer, num_records)
        elif mode=='val':
            return self.construct_dataset(self.validation_article, self.validation_question, self.validation_correct_answer, num_records)
        else:
            return self.construct_dataset(self.test_article, self.test_question, self.test_correct_answer, num_records)
        
