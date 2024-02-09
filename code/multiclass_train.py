import argparse
import torch
import numpy as np
import pandas as pd
import random
import evaluate
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding


# reproductivity 
random.seed(1)
np.random.seed(1)


# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variable
# 3 number of labels
TF_MAX_SIZE = 512

metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("f1")
metric_name = "f_macro"


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(classification_report(labels, predictions, digits=6))
    precision = metric1.compute(predictions=predictions, references=labels, average='weighted')["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average='weighted')["recall"]
    f_score = metric3.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    f_macro = metric3.compute(predictions=predictions, references=labels, average='macro')["f1"]
    return {"precision": precision, "recall": recall, "f_score": f_score, "f_macro": f_macro}


def train_model(transformer_model, train_df, dev_df, test_df,  model_name, num_label, label2id, id2label, entity, save_path):
    
    if model_name == 'RoBERTuito-uncased' or model_name == 'RoBERTuito-cased':
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        tokenizer.model_max_length = 128
    else: 
        tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length", truncation=True)
    
    # Tokenizer configuration for RoBERTuito
    # tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    # tokenizer.model_max_length = 128
    
    # For reproductivity 
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(transformer_model, num_labels=num_label, id2label=id2label, label2id=label2id)

    tokenized_train_dataset = tokenizer(train_df['tweet'].tolist(), truncation=True, padding=True)
    train_dataset = ClassificationDataset(tokenized_train_dataset, train_df.label.tolist())

    tokenized_eval_dataset = tokenizer(dev_df['tweet'].tolist(), truncation=True, padding=True)
    eval_dataset = ClassificationDataset(tokenized_eval_dataset, dev_df.label.tolist())
    
    tokenized_test_dataset = tokenizer(test_df['tweet'].tolist(), truncation=True, padding=True)
    test_dataset = ClassificationDataset(tokenized_test_dataset, test_df.label.tolist())
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    batch_train_size = 16
    batch_eval_size = 16
    training_args = TrainingArguments(
        output_dir="./log",
        # overwrite_output_dir=True,
        num_train_epochs=6,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_train_size,
        per_device_eval_batch_size=batch_eval_size,
        metric_for_best_model=metric_name,
        save_total_limit=1,
        seed = 1,
        learning_rate = 2e-5,
        weight_decay = 0.01
    )
    trainer = Trainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Salvamos el modelo reentrenado
    trainer.save_model(f'{save_path}/{model_name}_{entity}_finance_v2')
    tokenizer.save_pretrained(f'{save_path}/{model_name}_{entity}_finance_v2')
    
    print (f"PREDICCIONES SOBRE TEST {model_name} {entity}")
    predictions = trainer.predict (test_dataset)
    print(predictions.metrics)


def get_train_split(dataset, entity, label2id): 
    if entity == 'target': 
        dataset['target_sentiment'] = dataset['target_sentiment'].apply(lambda x: label2id[x])
        train_df = dataset[dataset['__split']=='train'][['tweet', 'target_sentiment']]
        val_df = dataset[dataset['__split']=='val'][['tweet', 'target_sentiment']]
        test_df = dataset[dataset['__split']=='test'][['tweet','target_sentiment']]
        
        train_df.rename(columns = {'target_sentiment':'label'}, inplace = True)
        val_df.rename(columns = {'target_sentiment':'label'}, inplace = True)
        test_df.rename(columns = {'target_sentiment':'label'}, inplace = True)
        
    elif entity == 'companies': 
        dataset['companies_sentiment'] = dataset['companies_sentiment'].apply(lambda x: label2id[x])
        train_df = dataset[dataset['__split']=='train'][['tweet', 'companies_sentiment']]
        val_df = dataset[dataset['__split']=='val'][['tweet', 'companies_sentiment']]
        test_df = dataset[dataset['__split']=='test'][['tweet','companies_sentiment']]
        
        train_df.rename(columns = {'companies_sentiment':'label'}, inplace = True)
        val_df.rename(columns = {'companies_sentiment':'label'}, inplace = True)
        test_df.rename(columns = {'companies_sentiment':'label'}, inplace = True)
        
    elif entity == 'consumers': 
        dataset['consumers_sentiment'] = dataset['consumers_sentiment'].apply(lambda x: label2id[x])
        train_df = dataset[dataset['__split']=='train'][['tweet', 'consumers_sentiment']]
        val_df = dataset[dataset['__split']=='val'][['tweet', 'consumers_sentiment']]
        test_df = dataset[dataset['__split']=='test'][['tweet','consumers_sentiment']]
        
        train_df.rename(columns = {'consumers_sentiment':'label'}, inplace = True)
        val_df.rename(columns = {'consumers_sentiment':'label'}, inplace = True)
        test_df.rename(columns = {'consumers_sentiment':'label'}, inplace = True)
        
   
    return train_df, val_df, test_df


def main(args):
    dataset_path = args.dataset_path
    save_path = args.save_path

    
    transformer_model = {'BETO-uncased': 'dccuchile/bert-base-spanish-wwm-uncased', 
                          'BETO-cased': 'dccuchile/bert-base-spanish-wwm-cased',
                          'MarIA': 'PlanTL-GOB-ES/roberta-base-bne',
                          'BERTIN': 'bertin-project/bertin-roberta-base-spanish',
                          'RoBERTuito-uncased': 'pysentimiento/robertuito-base-uncased',
                          'RoBERTuito-cased': 'pysentimiento/robertuito-base-cased',
                          'ALBETO': 'dccuchile/albert-base-spanish', 
                          'DistilBERT': 'dccuchile/distilbert-base-spanish-uncased', 
                          'XLM-RoBERTa': 'xlm-roberta-base'}
    
    
    # Read dataset                     
    dataset = pd.read_csv(dataset_path)
    
    label_list = sorted(dataset.target_sentiment.unique().tolist())
    print(label_list)
    num_label = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    entity = ['target', 'companies', 'consumers']
    
    for e in entity: 
        train_df, val_df, test_df = get_train_split(dataset, e, label2id)
        print(train_df)
        for k, v in transformer_model.items():
            train_model(v, train_df, val_df, test_df, k, num_label, label2id, id2label, e, save_path)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)