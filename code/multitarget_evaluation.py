import torch
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizerFast, AutoModelForCausalLM, MBartTokenizerFast, MBartTokenizer
from transformers import BartConfig, BartModel
from transformers import BartForConditionalGeneration, BartTokenizer, BartForCausalLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, pipeline
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sentiments = ['negative', 'positive', 'neutral']
# sentiments = ['negativo', 'positivo', 'neutral']

def check_response(class_list, response, text):
    if response in class_list: 
        return response
    else:
        print(f"{class_list} siendo token extraido {text}")
        return ""


def extract_triplets(text):
    triplets = []
    entity, target, companies, consumers  = '', '', '', ''
    
    text = text.strip()
    current = 'x'
    item = {}
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<absa>":
            current = 'e'
            entity = ''
        elif token == "<target>": 
            current = 't'
            target = ''
        elif token == "<companies>" or token == "<companies >": 
            current = 'c'
            companies = ''
        elif token == "<consumers>" or token == "<consumers >": 
            current = 's'
            consumers = ''
            
        else: 
            if current == 'e':
                entity += ' ' + token
                
            elif current == 't': 
                target += ' ' + token 
                
            elif current == 'c': 
                companies += ' ' + token 
                
            elif current == 's':
                consumers += ' ' + token
    
    item['target'] = entity.strip()
    item['target_sentiment'] = check_response(sentiments, target.strip(), text)
    item['companies_sentiment'] = check_response(sentiments, companies.strip(), text)
    item['consumers_sentiment'] = check_response(sentiments, consumers.strip(), text)

    # print(item)
    return item
    # return {'label': label.strip(), 'target': target.strip(), 'group': group.strip(), 'aggresive': aggresive.strip()}


def main(args): 
    dataset_path = args.dataset_path
    test_df = pd.read_csv(dataset_path)
   
    result_df = pd.DataFrame(columns=['tweet', 'target', 'target_sentiment', 'companies_sentiment', 'consumers_sentiment'])
    
    triplet_extractor = pipeline('text2text-generation', model='./t5_multitarget-checkpoints', tokenizer = './t5_multitarget-checkpoints', device=0)
    for index, row in test_df.iterrows(): 
    
        extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(row['tweet'], return_tensors=True, return_text=False)[0]["generated_token_ids"]])
        item_row = extract_triplets(extracted_text[0]) 
        item_row['tweet'] = row['tweet']
        
        result_df.loc[len(result_df)] = item_row
        
    result_df.to_csv(f'{args.save_path}/multitarget_result_prediction_e18.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/test.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)