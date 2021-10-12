from .embedding import *
import argparse
import ipdb
import joblib
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.utils.rnn import pad_sequence
import logging

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', default=32, type=int)
    return parser.parse_args()

def read_file(path):
    with open(path) as f:
        index, data = [], []
        for line in tqdm(f.readlines()):
            line = line.strip()
            idx, content = line.split('\t', 1)
            index.append(idx)
            data.append(content)
    return index, data

if __name__ == "__main__":
    args = vars(parser_args())
    modes = ['train', 'test', 'validation']
    datasets = ['dblp_cn', 'dblp_cv', 'dblp_ml', 'dblp_nlp']
    
    model = GloVe2Tensor()
    
    for dataset in datasets:
        for mode in modes:
            index, data = read_file(f'data/{dataset}/{mode}/content_{mode}')
            embeddings = model.convert(data)

        
            with open(f'data/{dataset}/{mode}/bert_embedding/embeddings_new', 'wb') as f:
                joblib.dump(embeddings, f)

            with open(f'data/{dataset}/{mode}/bert_embedding/indices_new', 'wb') as f:
                joblib.dump(index, f)
            print(f'[!] process {dataset}-{mode} over ...')
