## Harnessing Generative Text Techniques to Conquer Traditionally Extractive QA Challenges

An attempt to solve a traditionally Extractive QA task with Generative Text methods.

### Objective
Our objective is to fine-tune state-of-the-art Pretrained Language Models on the RACE (ReAding Comprehension from Examinations) dataset without providing multiple-choice answer options.

### Requirements - 
To download all package requirements.

```
pip install -r requirements.txt
```

### Setup - 
```
./setup.sh
```

### Training - 

Command line arguments: <br>

1. "base_model" - Base model to finetune <br>
2. "batch_size" - Batch size for the training <br>
3. "epochs" - Number of epochs to train <br>
4. "lr" - Learning rate
5. "workers" - Number of working units used to load the data
6. "device" - Device to be used for computations (cpu, cuda)
7. "max_input_length" - Maximum length of input text
8. "seed" - Seed for random initialization
9. "max_records_cut" - Fraction of records to train and validate on
10. "resume_from_epoch" - Resume from checkpoint @ specified epoch number
11. "evaluate" - Evalute (test set) model at epoch (Will skip training)
12. "num_test_records" - How many records to create the test set from?
13. "save_test_generation" - Save generated text from test evaluation


Command to train a base model with no optimizations<br>
```
python train.py --epochs=6 --base_model="t5-base" --batch_size=8
```

Command to train a base model with qLoRA<br>
```
python train_lora.py --epochs=4 --base_model="google/flan-t5-base" --batch_size=16
```

Command to train perform SFT training on base model with Unsloth and qLoRA optimizations<br>
```
python train_sft.py
```

### Evaluation on Test set - 

Evaluate a finetuned model on the test set from a specific epoch <br>
```
python train.py --base_model="t5-base" --batch_size=2 -e --resume_from_epoch=6
```

Evaluate a finetuned model with qLoRA on the test set from a specific epoch <br>
```
python train_lora.py --base_model="google/flan-t5-base" --batch_size=16 -e=2
```

### Inference/Testing - 

Evaluate the metrics on a smaller test set <br>
```
python inference.py
```
