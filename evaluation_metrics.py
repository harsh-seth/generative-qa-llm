#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
from rouge import Rouge
import datasets


# <h1> ROUGE </h1>

# In[5]:


hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)


# In[12]:


scores[0]


# <h1> BLEURT </h1>

# In[27]:


get_ipython().system('pip install git+https://github.com/google-research/bleurt.git')


# In[56]:


hypothesis = ["Delhi is the capital of India", "Delhi is the capital of India"]

reference = ["Delhi is capital of India", "Delhi is the capital of India"]
metric = datasets.load_metric("bleurt")
results = metric.compute(predictions = hypothesis, references = reference, )
print([round(v, 2) for v in results["scores"]])


# <h1> BERTScore </h1>

# In[ ]:


get_ipython().system('pip install bert_score')


# In[54]:


# Install the bert-score library if you haven't already
# !pip install bert-score

from bert_score import score

# Step 1: Prepare your reference and candidate sentences
references = ["Reference sentence 1", "Reference sentence 2"]
candidates = ["Candidate sentence 1", "Candidate sentence 2"]
context = ["Context 1", "Context 2"]

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


# In[ ]:




