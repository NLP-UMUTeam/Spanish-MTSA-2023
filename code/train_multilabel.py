import argparse
import torch
import numpy as np
import pandas as pd
import random
import evaluate
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
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
metric_name = "f1-macro"

target_names = ['target_pos', 'target_neu', 'target_neg',  'other_pos', 'other_neu', 'other_neg', 'consumer_pos', 'consumer_neu', 'consumer_neg']
sentiments = ['positive', 'neutral', 'negative']
sentiments2id = {label: i for i, label in enumerate(sentiments)}

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


def multi_label_metrics(predictions, labels, threshold=0.5):
    # Apply the sigmoid function to obtain results between 0 and 1
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # 1 when it exceeds 50% 
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # Apply the metric
    y_true = labels
    # Print the classification report  
    print(classification_report(y_true, y_pred, digits=4, target_names=target_names))
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    # Return as dictionary
    metrics = {'f1-macro': f1_macro_average,
               'f1-avg': f1_weighted_average,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


def get_label_format(num_label, target_sentiment, companies_sentiment, consumers_sentiment):
    labels = np.zeros(num_label)
    sep_idx = num_label / len(sentiments)
    
    labels[sentiments2id[target_sentiment]] = 1.0
    labels[sentiments2id[companies_sentiment] + 3] = 1.0 
    labels[sentiments2id[consumers_sentiment] + 6] = 1.0
    return labels.tolist()


def train_model(transformer_model, train_df, dev_df, model_name, num_label, label2id, id2label, save_path):
    
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
        return AutoModelForSequenceClassification.from_pretrained(transformer_model, num_labels=num_label, id2label=id2label, label2id=label2id, problem_type="multi_label_classification")

    tokenized_train_dataset = tokenizer(train_df['tweet'].tolist(), truncation=True, padding=True)
    train_dataset = ClassificationDataset(tokenized_train_dataset, train_df.labels.tolist())

    tokenized_eval_dataset = tokenizer(dev_df['tweet'].tolist(), truncation=True, padding=True)
    eval_dataset = ClassificationDataset(tokenized_eval_dataset, dev_df.labels.tolist())
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    batch_train_size = 8
    batch_eval_size = 8
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
    trainer.save_model(f'{save_path}/{model_name}_finance')
    tokenizer.save_pretrained(f'{save_path}/{model_name}_finance')


def main(args):
    dataset_path = args.dataset_path
    save_path = args.save_path

    dataset = pd.read_csv(dataset_path)
    dataset = dataset[['tweet', '__split', 'target', 'target_sentiment', 'companies_sentiment', 'consumers_sentiment']]
    
    train_df = dataset[dataset['__split']=='train']
    train_df.to_csv(f"dataset/train.csv", index=False)
    val_df = dataset[dataset['__split']=='val']
    val_df.to_csv(f"dataset/val.csv", index=False)
    test_df = dataset[dataset['__split']=='test']
    test_df.to_csv(f"dataset/test.csv", index=False)
    
    print(target_names)
    num_label = len(target_names)
    label2id = {label: i for i, label in enumerate(target_names)}
    id2label = {i: label for i, label in enumerate(target_names)}
    
    dataset['labels'] = dataset.apply(lambda row: get_label_format(num_label, row['target_sentiment'], row['companies_sentiment'], row['consumers_sentiment']), axis=1)
    
    train_df = dataset[dataset['__split']=='train'][['labels', 'tweet']]
    val_df = dataset[dataset['__split']=='val'][['labels', 'tweet']]
    test_df = dataset[dataset['__split']=='test'][['labels', 'tweet']]
    
    
    transformer_model = {'BETO-uncased': 'dccuchile/bert-base-spanish-wwm-uncased', 
                         'BETO-cased': 'dccuchile/bert-base-spanish-wwm-cased',
                         'MarIA': 'PlanTL-GOB-ES/roberta-base-bne',
                         'BERTIN': 'bertin-project/bertin-roberta-base-spanish',
                         'RoBERTuito-uncased': 'pysentimiento/robertuito-base-uncased',
                         'RoBERTuito-cased': 'pysentimiento/robertuito-base-cased',
                         'ALBETO': 'dccuchile/albert-base-spanish', 
                         'DistilBERT': 'dccuchile/distilbert-base-spanish-uncased', 
                         'XLM-RoBERTa': 'xlm-roberta-base'}
     
        
        
    for k, v in transformer_model.items():
        train_model(v, train_df, val_df, k, num_label, label2id, id2label, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)