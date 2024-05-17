import os
from tqdm import tqdm
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from torch.utils.data import DataLoader
import argparse

from data.RACE_Dataset import RaceDataset
from utils.data_utils import dataset_collator
from utils.evaluation_metrics import get_eval_scores

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training T5 T2T model')

    parser.add_argument('--t5_model', type=str, default="t5-base",
                        help="What type of T5 model do you want use? (default: 't5-base')")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs (default: 2)')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (Adam) (default: 1e-4)')

    parser.add_argument('--workers', type=int, default=0,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')

    parser.add_argument('--max_records_cut', type=float, default=1.0,
                    help='Fraction of records to train and validate on (range: 0.0 - 1.0, default: 1.0 - i.e. all records)')

    parser.add_argument('--resume_from_epoch', type=int, default=None,
                help='Resume from checkpoint @ specified epoch number')
    
    parser.add_argument('--evaluate', '-e', action="store_true",
                help='Skip training?')

    parser.add_argument('--num_test_records', type=int, default=None,
                help='How many records to create the test set from? (default: None = All records)')
    
    parser.add_argument('--save_test_generation', action="store_true",
                help='Save generated text from test evaluation?')

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
    f1_score, exact_match_score = get_eval_scores(tokenizer, model_predictions_encoded, target_encoded)
    return f1_score, exact_match_score, model_predictions_encoded

def train(model, tokenizer, optimizer, train_dataloader, validation_dataloader, num_train_epochs, device, max_input_length = 512, starting_epoch = 0, save_path_prefix = "results/t5-pretrained"):
    best_f1_score: int = 0
    for epoch in range(starting_epoch, num_train_epochs):
        print(f"\n[Epoch {epoch + 1} / {num_train_epochs}]")

        epoch_train_loss = trainOnce(model, tokenizer, optimizer, train_dataloader, max_input_length, device)
        epoch_train_loss /= len(train_dataloader.dataset)
        print(f"\t Train loss = {epoch_train_loss:.4f}")

        model.save_pretrained(f'{save_path_prefix}/model/checkpoint-{epoch+1}')
        tokenizer.save_pretrained(f'{save_path_prefix}/tokenizer/checkpoint-{epoch+1}')

        with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
            results_file.write(f"train,{len(train_dataloader.dataset)},,,{epoch+1},{epoch_train_loss:.4f}\n")

        f1_score, exact_match_score, _ = evaluateOnce(model, tokenizer, validation_dataloader, max_input_length, device)
        print(f"\t Validation F1 = {f1_score:.2f}, Exact Match (EM) = {exact_match_score:.2f}")
        
        with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
            results_file.write(f"validation,{len(validation_dataloader.dataset)},{f1_score:.2f},{exact_match_score:.2f},{epoch+1}\n")

        if f1_score > best_f1_score :
            model.save_pretrained(f'{save_path_prefix}/model/best-f1')
            tokenizer.save_pretrained(f'{save_path_prefix}/tokenizer/best-f1')
            best_f1_score = f1_score

    model.save_pretrained(
        f'{save_path_prefix}/model/checkpoint-{epoch+1}')
    tokenizer.save_pretrained(
        f'{save_path_prefix}/tokenizer/checkpoint-{epoch+1}')


if __name__ == '__main__':
    args = parse_command_line_arguments()
    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    # Set seed
    set_seed(args.seed)
    save_path_prefix = f"results/{args.t5_model}"
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    model = AutoModelForSeq2SeqLM.from_pretrained(f"{save_path_prefix}/model/checkpoint-{args.resume_from_epoch}" if args.resume_from_epoch else args.t5_model)
    tokenizer = AutoTokenizer.from_pretrained(f"{save_path_prefix}/tokenizer/checkpoint-{args.resume_from_epoch}" if args.resume_from_epoch else args.t5_model)
    model.to(args.device)
    # creating the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    raceDataset = RaceDataset()
    train_set = raceDataset.get_dataset('train', num_records=int(80292*args.max_records_cut))
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: dataset_collator(data, preprocessed=True))
    
    validation_set = raceDataset.get_dataset('val', num_records=int(4495*args.max_records_cut))
    validation_dataloader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: dataset_collator(data, preprocessed=True))

    test_set = raceDataset.get_dataset('test', num_records=args.num_test_records)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=lambda data: dataset_collator(data, preprocessed=True))

    if not args.evaluate:
        train(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            num_train_epochs=args.epochs, 
            device=args.device, 
            starting_epoch=args.resume_from_epoch if args.resume_from_epoch else 0,
            save_path_prefix=save_path_prefix
        )

    print("\n[Evaluation]")
    f1_score, exact_match_score, model_pred_tokens = evaluateOnce(model, tokenizer, test_dataloader, args.max_input_length, args.device)
    print(f"\t Evaluation F1 = {f1_score:.2f}, Exact Match (EM) = {exact_match_score:.2f}")
    with open(f"{save_path_prefix}/metrics.csv", 'a') as results_file:
        results_file.write(f"test,{len(test_dataloader.dataset)},{f1_score:.2f},{exact_match_score:.2f}\n")
    
    if args.save_test_generation:
        with open('outputs.csv', 'a') as file:
            preds = tokenizer.batch_decode(model_pred_tokens, skip_special_tokens=True)
            for pred in preds:
                file.write(f"{pred}\n")
