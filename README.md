## Harnessing Generative Text Techniques to Conquer Traditionally Extractive QA Challenges

An attempt to solve a traditionally Extractive QA task with Generative Text methods.

### Objective
Our objective is to fine-tune state-of-the-art Pretrained Language Models on the RACE (ReAding Comprehension from Examinations)dataset without providing multiple-choice answer options.

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

1. "t5_model" - T5 model to finetune <br>
2. "batch_size" - Batch size for the training <br>
3. "epochs" - Number of epochs to train <br>
4. "lr" - Learning rate
5. "workers" - Number of working units used to load the data
6. "device" - Device to be used for computations (cpu, cuda)
7. "max_input_length" - Maximum length of input text
8. "seed" - Seed for random initialization
9. "max_records_cut" - Fraction of records to train and validate on
10. "evaluate_at_epoch" - Resume from checkpoint @ specified epoch number


Command to train the T5-base mode -
'''
python train.py --epochs=6 --t5_model="t5-base" --batch_size=8
'''
<br>
Command to train the flan-t5-base with qLoRA
```
python train_lora.py --epochs=4 --t5_model="google/flan-t5-base" --batch_size=16 -e=2
```
<br>
Command to train the instruction fine tuned
```
python train_sft.py
```

### Inference/Testing - 

Command to run the model on test set
'''
python inference.py
'''