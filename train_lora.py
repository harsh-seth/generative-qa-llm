import argparse
import os
import torch

from peft import get_peft_model, PeftConfig, prepare_model_for_kbit_training, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, T5Tokenizer, set_seed

from data.Dataset import Dataset
from data.RACE_Dataset import RaceDataset

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='CLI for fine-tuning a T5 Text-to-Text model')

    parser.add_argument('--t5_model', type=str, default="google/flan-t5-base",
                        help="What type of T5 model do you want use? (default: 'google/flan-t5-base')")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs (default: 2)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=0,
                        help='number of working units used to load the data (default: 0)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cuda)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum length of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random initialization (default: 42)')
    
    parser.add_argument('--max_records_cut', type=float, default=1.0,
                    help='Fraction of records to train and validate on (range: 0.0 - 1.0, default: 1.0 - i.e. all records)')
    
    parser.add_argument('--evaluate_at_epoch', '-e', type=int, default=None,
                help='Evaluate model at checkpoint @ specified epoch number')
    
    parsed_arguments = parser.parse_args()

    return parsed_arguments


def trainOnce(model, tokenizer, optimizer, dataloader, max_input_length, device):
    model.train()
    train_loss = 0.
    for prompts, targets in tqdm(dataloader):
        optimizer.zero_grad()
        encoded_inputs = tokenizer(
            prompts,
            padding="longest",
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded_targets = tokenizer(
            targets,
            padding="longest",
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
        encoded_targets = encoded_targets.input_ids

        # replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
        encoded_targets[encoded_targets == tokenizer.pad_token_id] = -100

        input_ids = input_ids.to(device)
        encoded_targets = encoded_targets.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=encoded_targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(prompts)
    return train_loss

def evaluateOnce(model, tokenizer, dataloader, max_input_length, device):
    model.eval()
    model_predictions_encoded = []
    target_encoded = []
    with torch.no_grad():
        for prompts, targets in tqdm(dataloader):
            encoded_inputs = tokenizer(
                prompts,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                targets,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            encoded_inputs = encoded_inputs.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            model_predictions = model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
    f1_score, exact_match_score = dataloader.dataset.evaluate(model_predictions_encoded, target_encoded)
    return f1_score, exact_match_score

def train(model, tokenizer, optimizer, train_dataloader, validation_dataloader, num_train_epochs, device, max_input_length = 512, starting_epoch = 0, save_path_prefix = "results/t5-pretrained"):
    best_f1_score: int = 0
    for epoch in range(starting_epoch, num_train_epochs):
        print(f"\n[Epoch {epoch + 1} / {num_train_epochs}]")

        epoch_train_loss = trainOnce(model, tokenizer, optimizer, train_dataloader, max_input_length, device)
        epoch_train_loss /= len(train_dataloader.dataset)
        print(f"\t Train loss = {epoch_train_loss:.4f}")

        model.save_pretrained(f'{save_path_prefix}/checkpoint-{epoch+1}/model')
        tokenizer.save_pretrained(f'{save_path_prefix}/checkpoint-{epoch+1}/tokenizer')

        with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
            results_file.write(f"train,{len(train_dataloader.dataset)},,,{epoch+1},{epoch_train_loss:.4f}\n")

        f1_score, exact_match_score = evaluateOnce(model, tokenizer, validation_dataloader, max_input_length, device)
        print(f"\t Validation F1 = {f1_score:.2f}, Exact Match (EM) = {exact_match_score:.2f}")
        
        with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
            results_file.write(f"validation,{len(validation_dataloader.dataset)},{f1_score:.2f},{exact_match_score:.2f},{epoch+1}\n")

        if f1_score > best_f1_score :
            model.save_pretrained(f'{save_path_prefix}/best-f1/model')
            tokenizer.save_pretrained(f'{save_path_prefix}/best-f1/tokenizer')
            best_f1_score = f1_score

    model.save_pretrained(
        f'{save_path_prefix}/checkpoint-{epoch+1}/model')
    tokenizer.save_pretrained(
        f'{save_path_prefix}/checkpoint-{epoch+1}/tokenizer')


if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)

    save_path_prefix = f"results/{args.t5_model}"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.t5_model, quantization_config=bnb_config, device_map="auto")
    peft_config = None

    # Can either train the model, or evaluate it
    if args.evaluate_at_epoch:
        peft_config = PeftConfig.from_pretrained(f"{save_path_prefix}/checkpoint-{args.evaluate_at_epoch}/model")
    else:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    tokenizer = T5Tokenizer.from_pretrained(f"{save_path_prefix}/checkpoint-{args.evaluate_at_epoch}/tokenizer" if args.evaluate_at_epoch else args.t5_model)

    raceDataset = RaceDataset()
    train_set = raceDataset.get_dataset('train', num_records=int(80292*args.max_records_cut))
    train_set = Dataset(train_set, tokenizer)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: train_set.pack_minibatch(data))
    
    validation_set = raceDataset.get_dataset('val', num_records=int(4495*args.max_records_cut))
    validation_set = Dataset(validation_set, tokenizer)
    validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: validation_set.pack_minibatch(data))

    test_set = raceDataset.get_dataset('test')
    test_set = Dataset(test_set, tokenizer)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: test_set.pack_minibatch(data))

    if args.evaluate_at_epoch:
        print("\n[Evaluation]")
        f1_score, exact_match_score = evaluateOnce(model, tokenizer, test_dataloader, args.max_input_length, args.device)
        print(f"\t Evaluation F1 = {f1_score:.2f}, Exact Match (EM) = {exact_match_score:.2f}")
        with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
            results_file.write(f"test,{len(test_dataloader.dataset)},{f1_score:.2f},{exact_match_score:.2f}\n")

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            num_train_epochs=args.epochs, 
            device=args.device, 
            starting_epoch=0,
            max_input_length=args.max_input_length,
            save_path_prefix=save_path_prefix,
        )
