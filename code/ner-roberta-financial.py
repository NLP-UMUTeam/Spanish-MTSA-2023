import argparse
import torch
import string
import numpy as np
import pandas as pd
import random
from datasets import load_metric
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, AutoTokenizer
from transformers import RobertaForTokenClassification, XLMRobertaForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import set_seed
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# reproductilibty
set_seed(1)
random.seed(1)
np.random.seed(1)
# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tag = ['O', 'B-TARGET', 'I-TARGET']
tag2idx = {t: i for i, t in enumerate(tag)}

# Global variable
NUM_LABELS = len(tag)

metric_name = "f1"
default_weight_decay = 0.01


def compute_metrics(p):
    metric = load_metric("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_predictions = [[tag[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[tag[l] for l in label if l != -100] for label in labels]
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Obtenemos informe de clasificacion
    print(classification_report(true_labels, true_predictions, mode = 'strict', scheme = IOB2, digits = 6))

    # Retornamos los resultados obtenidos
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}



def train_model(transformer_model, train_df, dev_df, test_df, model_name, save_path):
    def model_init(trial):
        if model_name == 'XLM-RoBERTa':
            return XLMRobertaForTokenClassification.from_pretrained(transformer_model, num_labels=NUM_LABELS)
        else: 
            return RobertaForTokenClassification.from_pretrained(transformer_model, num_labels=NUM_LABELS)

    if model_name == 'XLM-RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    elif model_name == 'RoBERTuito-uncased' or model_name == 'RoBERTuito-cased': 
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
    else: 
        tokenizer = RobertaTokenizer.from_pretrained(transformer_model)
        
    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_tokenized_datasets, eval_tokenized_datasets, test_tokenized_datasets = get_tokenizer_dataset_roberta(tokenizer,train_df, dev_df, test_df, model_name)

    batch_train_size = 16
    batch_eval_size = 16

    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=batch_train_size,
        per_device_eval_batch_size=batch_eval_size,
        metric_for_best_model=metric_name,
        weight_decay=0.01,
        seed=1,
        learning_rate=2e-5,
    )

    trainer = Trainer(
        model_init=model_init,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=eval_tokenized_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Salvamos el modelo reentrenado
    trainer.save_model(f'{save_path}/{model_name}_finance_ner')
    tokenizer.save_pretrained(f'{save_path}/{model_name}_finance_ner')
    
    print (f"PREDICCIONES SOBRE TEST {model_name}")
    predictions = trainer.predict (test_tokenized_datasets)
    print(predictions.metrics)


def clean_text(token):
    token_clean = "".join([i for i in token if i not in string.punctuation])
    token_clean = "".join([i for i in token_clean if i != '¿' and i != '¡'])
    return token_clean.lower()


def parser_text(dataset):
    id = 1
    df_result = pd.DataFrame(columns=['Sentence_id', 'Token', 'Target'])
    # for texto in dataset['Tweet']:
    for idx, row in dataset.iterrows():
        texto = clean_text(row['tweet'])
        target_list = row['target'].split(' ')
        target_list = [clean_text(x) for x in target_list]
        occur = 0
        for token in texto.split(' '):
            # print(id)
            if token in target_list:
                if occur == 0:
                    row = {'Sentence_id': id, 'Token': token, 'Target': 'B-TARGET'}
                    # df_result = df_result.append(row, ignore_index=True)
                    df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
                    occur += 1
                else:
                    row = {'Sentence_id': id, 'Token': token, 'Target': 'I-TARGET'}
                    # df_result = df_result.append(row, ignore_index=True)
                    df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
                    occur += 1
            else:
                row = {'Sentence_id': id, 'Token': token, 'Target': 'O'}
                # df_result = df_result.append(row, ignore_index=True)
                df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
        id = id + 1
    df_result['clean_token'] = df_result['Token'].apply(lambda x: clean_text(x))
    df_result = create_df(df_result)
    return df_result


class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        # agg_func = lambda s: [(w, t) for w, t in zip(s["Token_clean"].astype(str).values.tolist(),s["Punc_tag"].values.tolist())]
        agg_func = lambda s: [(w, t) for w, t in zip(s["clean_token"].astype(str).values.tolist(),
                                                     s["Target"].values.tolist())]

        # Por frases
        self.grouped = self.dataset.groupby("Sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def create_df(dataset):
    getter = SentenceGetter(dataset)
    sentences = [[s[0] for s in sent] for sent in getter.sentences]
    targets = [[s[1] for s in sent] for sent in getter.sentences]
    labels = [[tag2idx.get(p) for p in punc] for punc in targets]
    df = pd.DataFrame({'tokens': sentences, 'target': targets, 'labels': labels})
    return df


def adjust_label(input_to_tokens, labels):
    l = []
    labels_id = 0
    for idx in range(len(input_to_tokens)):
        token_id = input_to_tokens[idx]
        if input_to_tokens[idx] == '<s>' or input_to_tokens[idx] == '</s>':
            l.append(-100)
        elif input_to_tokens[idx][0] == 'Ġ':
            l.append(labels[labels_id])
            labels_id = labels_id + 1
        else:
            l.append(labels[labels_id - 1])
    return l


def adjust_label_xlm(input_to_tokens, labels):
    l = []
    labels_id = 0
    for idx in range(len(input_to_tokens)):
        token_id = input_to_tokens[idx]
        if input_to_tokens[idx] == '<s>' or input_to_tokens[idx] == '</s>':
            l.append(-100)
        elif input_to_tokens[idx][0] == '▁':
            if labels_id >= len(labels): 
              l.append(labels[len(labels)-1])
            else:
              l.append(labels[labels_id])
              labels_id = labels_id + 1
        else:
              
            l.append(labels[labels_id - 1])
    return l
    

def get_tokenizer_dataset_roberta(tokenizer, train_dataset, eval_dataset, test_dataset, type_model):
    def tokenize_and_labels(examples):
        encodings = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

        input_to_tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'])
        if type_model == "XLM-RoBERTa": 
            labels = adjust_label_xlm(input_to_tokens, examples['labels'])
        else:
            labels = adjust_label(input_to_tokens, examples['labels'])

        return {**encodings, 'labels': labels}

    train_tokenized_datasets = train_dataset.map(tokenize_and_labels)
    eval_tokenized_datasets = eval_dataset.map(tokenize_and_labels)
    test_tokenized_datasets = test_dataset.map(tokenize_and_labels)
    return train_tokenized_datasets, eval_tokenized_datasets, test_tokenized_datasets
    



def create_dataset(df_train, df_eval, df_test):
    # Parseamos el conjunto de datos en formato Dataset a través de la librería datasets
    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)
    test_dataset = Dataset.from_pandas(df_test)
    return train_dataset, eval_dataset, test_dataset


def main(args):
    train_path = args.train_path
    save_path = args.save_path

    financial_dataset = pd.read_csv(train_path)

    news_train_df = financial_dataset.loc[(financial_dataset['__split'] == 'train')]
    news_eval_df = financial_dataset.loc[(financial_dataset['__split'] == 'val')]
    news_test_df = financial_dataset.loc[(financial_dataset['__split'] == 'test')]


    # Parser the df to token classification format
    dataset_news_train = parser_text(news_train_df)
    dataset_news_train.to_csv("check.csv")
    dataset_news_eval = parser_text(news_eval_df)
    dataset_news_test = parser_text(news_test_df)
    
    dataset_all_train, dataset_all_eval, test_dataset = create_dataset(dataset_news_train, dataset_news_eval, dataset_news_test)
    

    models = { "MarIA": "PlanTL-GOB-ES/roberta-base-bne", 
               "Bertin": "bertin-project/bertin-roberta-base-spanish",  
               'XLM-RoBERTa': 'xlm-roberta-base', 
               'RoBERTuito-uncased': 'pysentimiento/robertuito-base-uncased',
               'RoBERTuito-cased': 'pysentimiento/robertuito-base-cased',
              }


        
    for k, v in models.items(): 
        train_model(v, dataset_all_train, dataset_all_eval, test_dataset, k, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='./dataset.csv')

    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)
