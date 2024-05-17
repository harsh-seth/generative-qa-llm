import torch
from datasets import load_dataset
from trl import SFTTrainer
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from tqdm import tqdm

from data.Dataset import get_eval_scores

def evaluateOnce(model, tokenizer, dataloader):
    model.eval()
    model_predictions_encoded = []
    target_encoded = []
    with torch.no_grad():
        for prompts, targets in tqdm(dataloader):
            encoded_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded_targets = tokenizer(
                targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded_targets = encoded_targets.input_ids
            print("transferring to device...")
            encoded_inputs = encoded_inputs.to("cuda")
            encoded_targets = encoded_targets.to("cuda")
            print("starting generation...")
            model_predictions = model.generate(**encoded_inputs, max_new_tokens = 512, use_cache=True)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1_score, exact_match_score = get_eval_scores(model_predictions_encoded, target_encoded)
    return f1_score, exact_match_score

# Instantiate FastLanguageModel
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
prompt_template = """Answer the following question based on the provided context
## Question:
{}

## Context:
{}

## Response:
{}
"""
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    options = examples["options"]
    answers = examples["answer"]
    contexts = examples["article"]
    questions = examples["question"]

    # # removing questions which depend on the options
    # regex = r"^which of the following|are correct except|^the following are true according to the passage except"
    # for idx, question in enumerate(questions):
    #     if re.match(regex, question.lower()):
    #         options.pop(idx)
    #         answers.pop(idx)
    #         contexts.pop(idx)
    #         questions.pop(idx)
    
    responses = []
    option_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    for idx, answer in enumerate(answers):
        responses.append(options[idx][option_index[answer]])

    texts = []
    for i,j,k  in zip(questions, contexts, responses):
        text = prompt_template.format(i,j,k) + EOS_TOKEN
        texts.append(text)
    return {"texts": texts}

train_dataset = load_dataset("ehovy/race", "all", split="train")
validation_dataset = load_dataset("ehovy/race", "all", split="validation")
test_dataset = load_dataset("ehovy/race", "all", split="test")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
validation_dataset = validation_dataset.map(formatting_prompts_func, batched=True)

def collate_fn(examples):
    prompts = []
    responses = []
    option_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

    for example in examples:
        option = example["options"]
        answer = example["answer"]
        context = example["article"]
        question = example["question"]
        responses.append(option[option_index[answer]])
        prompts.append(prompt_template.format(question, context, ""))

    return prompts, responses
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    max_seq_length=max_seq_length,
    dataset_text_field="texts",
    dataset_num_proc=4,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 4,
        max_steps=100,
        # num_train_epochs = 2,
        warmup_steps = 5,
        # save_steps=0.5,
        learning_rate = 0.0005,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        evaluation_strategy="steps",
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
trainer.train()

print("\n[Evaluation]")
FastLanguageModel.for_inference(model)
f1_score, exact_match_score = evaluateOnce(model, tokenizer, test_dataloader)
print(f"\t Evaluation F1 = {f1_score:.2f}, Exact Match (EM) = {exact_match_score:.2f}")
with open(f"outputs/metrics.csv", 'w+') as results_file:
    results_file.write(f"test,{len(test_dataloader.dataset)},{f1_score:.2f},{exact_match_score:.2f}\n")
