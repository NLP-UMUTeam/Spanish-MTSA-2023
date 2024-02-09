import torch
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizerFast, AutoModelForCausalLM, MBartTokenizerFast, MBartTokenizer
from transformers import BartConfig, BartModel
from transformers import BartForConditionalGeneration, BartTokenizer, BartForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
label_pad_token_id = -100

sentiments = {'negative':'negativo', 'positive':'positivo', 'neutral':'neutral'}

def preprocess_function(examples, tokenizer):
    inputs = examples['tweet']
    targets = examples['labels']
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    # Basicamente, cuando usamos padding que sea max_length, los input_ids demás se codifica en pad_token_id
    # Por lo que si assignamos a -100, lo podremos ignorar durante en el entrenamiento
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args): 
        
    dataset_path = args.dataset_path
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[['tweet', '__split', 'target', 'target_sentiment', 'companies_sentiment', 'consumers_sentiment']]
    
    train_df = dataset[dataset['__split']=='train']
    train_df.to_csv(f"dataset/train.csv", index=False)
    val_df = dataset[dataset['__split']=='val']
    val_df.to_csv(f"dataset/val.csv", index=False)
    test_df = dataset[dataset['__split']=='test']
    test_df.to_csv(f"dataset/test.csv", index=False)
    
    # Add labels column
    train_df['labels'] = train_df.apply(lambda row: f"<absa> {row['target']} <target> {row['target_sentiment']} <companies> {row['companies_sentiment']} <consumers> {row['consumers_sentiment']}", axis=1)
    val_df['labels'] = val_df.apply(lambda row: f"<absa> {row['target']} <target> {row['target_sentiment']} <companies> {row['companies_sentiment']} <consumers> {row['consumers_sentiment']}", axis=1)
    
        
    if args.model == "mbart": 
        model_name = "facebook/mbart-large-50"
    elif args.model == "mt5":
        model_name = "google/mt5-base"
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
    # Import the model 
    config = AutoConfig.from_pretrained(
        model_name,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        forced_bos_token_id=None,
    )


    model = AutoModelForSeq2SeqLM.from_pretrained(
           model_name,
            config=config
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)    
    
    # Define the train and eval dataset
    train_df = Dataset.from_pandas(train_df)
    train_dataset = train_df.map(
            preprocess_function,
            batched=True,
            num_proc=4, 
            fn_kwargs={"tokenizer": tokenizer}
        )

    val_df = Dataset.from_pandas(val_df)
    dev_dataset = val_df.map(
            preprocess_function,
            batched=True,
            num_proc=4, 
            fn_kwargs={"tokenizer": tokenizer}
        )
        
    print(train_dataset['labels'][0])
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=label_pad_token_id)
    
    # Train the model
    
    arguments = Seq2SeqTrainingArguments(output_dir=args.save_path,
                        do_train=True,
                        do_eval=True,
                        save_total_limit=1,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=4,
                        per_device_eval_batch_size=4,
                        learning_rate=2e-5,
                        weight_decay=0.01,
                        num_train_epochs=24)
    
    
    trainer = Seq2SeqTrainer(model=model,
                    args=arguments,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset)
    
    
    # Training time
    trainer.train()
    trainer.save_model(f"{args.save_path}/{model_name}_multitarget-checkpoints_e24")
    tokenizer.save_pretrained(f"{args.save_path}/{model_name}_multitarget-checkpoints_e24")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset.csv')
    parser.add_argument('--model', type=str, default='mbart')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)