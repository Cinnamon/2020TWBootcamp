import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import ast
import tqdm

from transformers import AutoConfig, AutoTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset


class SashimiDataset(Dataset):
    def __init__(self, data):
        super(SashimiDataset, self).__init__()
        self.text = data.Text
        self.label = data.Label
        
    def __len__(self):
        return self.text.shape[0]*4
    
    def __getitem__(self, idx):
        text = self.text[idx // len(QUESTIONS)]
        qs_idx = idx % len(QUESTIONS)
        label = self.label[idx // len(QUESTIONS)][qs_idx]
        question = QUESTIONS[qs_idx]
        
        inputs = tokenizer(question, text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        label = torch.LongTensor(label)
        
        plus_idx = (inputs['input_ids'] == 102).nonzero()[0,1]
        label += plus_idx + 1
        
        return inputs, label
        
        
def get_data(path, sep=',', index_col=None):
    data = pd.read_csv(path, sep=sep, index_col=index_col)
    data['Text'] = [ast.literal_eval(data.Text[i])[0] for i in range(data.shape[0])]
    data['Label'] = [ast.literal_eval(data.Label[i]) for i in range(data.shape[0])]
    return data

if __name__ == '__main__':
    
    config = AutoConfig.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2", use_fast=True)
    
    QUESTIONS = [
        'What activity?',
        'What date?',
        'What time?',
        'Where to go?'
    ]

    data = get_data('./data/training_data.csv', sep='\t')
    dataset = SashimiDataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 2e-5)

    for epoch in range(1):
        total_loss = 0
        
        for inputs, label in tqdm.tqdm(dataloader):
            # Reform the inputs
            for k in inputs.keys():
                inputs[k] = inputs[k].squeeze(1)

            optimizer.zero_grad()
            outputs = model(**inputs)
            start, end = outputs[0], outputs[1]

            loss = loss_fn(start, label[:,0]) + loss_fn(end, label[:,1])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Train loss {total_loss}')
