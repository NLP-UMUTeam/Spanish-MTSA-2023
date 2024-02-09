import pandas as pd
import argparse
import torch
import ast
import numpy as np
import pandas as pd
import random
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# reproductivity 
random.seed(1)
np.random.seed(1)
TF_MAX_SIZE = 512

# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_names = ['target_pos', 'target_neu', 'target_neg',  'other_pos', 'other_neu', 'other_neg', 'consumer_pos', 'consumer_neu', 'consumer_neg']
sentiments = ['positive', 'neutral', 'negative']
sentiments2id = {label: i for i, label in enumerate(sentiments)}


def get_label_format(num_label, target_sentiment, companies_sentiment, consumers_sentiment):
    labels = np.zeros(num_label)
    sep_idx = num_label / len(sentiments)
    
    labels[sentiments2id[target_sentiment]] = 1.0
    labels[sentiments2id[companies_sentiment] + 3] = 1.0 
    labels[sentiments2id[consumers_sentiment] + 6] = 1.0
    return labels.tolist()
    
    
def get_tokenizer_eval_dataset(tokenizer, eval_dataset):
    def preprocess_data(examples):
        # take a batch of texts
        text = examples["tweet"]
        # encode them
        encoding = tokenizer(text, padding=True, truncation=True)
        return encoding

    eval_tokenized_datasets = eval_dataset.map(preprocess_data, batched=True)
    return eval_tokenized_datasets
    
    

def eval_model(transformer_model, model_name, dev_df, num_labels, save_path):
    
    if model_name == 'RoBERTuito-uncased' or model_name == 'RoBERTuito-cased': 
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        tokenizer.model_max_length = 128
    else:
        tokenizer = AutoTokenizer.from_pretrained(transformer_model, max_length=TF_MAX_SIZE, padding="max_length", truncation=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(transformer_model, num_labels=num_labels, problem_type="multi_label_classification").to(device)

    eval_dataset = Dataset.from_pandas(dev_df)
    eval_tokenized_datasets = get_tokenizer_eval_dataset(tokenizer, eval_dataset)

    threshold = 0.5
    y_predictions = []
    y_labels = [] 
    with torch.no_grad():
        for idx in range(0, len(eval_tokenized_datasets)):
            predictions = []
            input_ids = torch.tensor(eval_tokenized_datasets[idx]['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(eval_tokenized_datasets[idx]['attention_mask']).unsqueeze(0).to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(outputs.cpu()))
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= threshold)] = 1
            y_pred = y_pred[0].tolist()
            y_predictions.append(y_pred)

   
    print(classification_report(eval_tokenized_datasets['labels'], y_predictions, digits=6))

def main(args):
    dev_path = args.dev_path
    save_path = args.save_path

    dev_df = pd.read_csv(dev_path)
    dev_df = dev_df[['tweet', '__split', 'target', 'target_sentiment', 'companies_sentiment', 'consumers_sentiment']]
    
    num_labels = len(target_names)    
    dev_df['labels'] = dev_df.apply(lambda row: get_label_format(num_labels, row['target_sentiment'], row['companies_sentiment'], row['consumers_sentiment']), axis=1)
    
    label2id = {label: i for i, label in enumerate(target_names)}
    id2label = {i: label for i, label in enumerate(target_names)}
    
    transformer_model = { 'BETO-uncased': 'results/BETO-uncased_finance', 
                          'BETO-cased': 'results/BETO-cased_finance',
                          'MarIA': 'results/MarIA_finance',
                          'BERTIN': 'results/BERTIN_finance',
                          'RoBERTuito-uncased': 'results/RoBERTuito-uncased_finance',
                          'RoBERTuito-cased': 'results/RoBERTuito-cased_finance',
                          'ALBETO': 'results/ALBETO_finance', 
                          'DistilBERT': 'results/DistilBERT_finance', 
                          'XLM-RoBERTa': 'results/XLM-RoBERTa_finance'
                        }
    
    for k, v in transformer_model.items():                      
        eval_model(v, k, dev_df, num_labels, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_path', type=str, default='./dataset/test.csv') 
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)
