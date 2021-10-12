from .embedding import *
import argparse
import ipdb
import pickle
import joblib
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import logging
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--min_length', default=50, type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--opt_level', default='O2', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fine-tune', dest='fine_tune', action='store_true')
    parser.add_argument('--no-fine-tune', dest='fine_tune', action='store_false')
    parser.add_argument('--pool', dest='pool', action='store_true')
    parser.add_argument('--no-pool', dest='pool', action='store_false')
    return parser.parse_args()

def read_file(path):
    with open(path) as f:
        data = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            index, content = line.split('\t', 1)
            data.append((index, content))
    return data

# NOTE:
def load_model(model, path):
    state_dict = torch.load(path)
    try:
        model.load_state_dict(state_dict)
    except:
        current_module = True if 'module' in [i[0] for i in model.state_dict().items()][0] else False
        saved_module = True if 'module' in [i[0] for i in state_dict.items()][0] else False
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if saved_module and not current_module:
                name = k[7:]
                new_state_dict[name] = v
            elif not saved_module and current_module:
                name = f"module.{k}"
                new_state_dict[name] = v
            else:
                pass
        model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path} over ...')

class dblpDataset(Dataset):

    def __init__(self, data, max_length=256, min_length=50):
        super(dblpDataset, self).__init__()
        self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = data
        self.max_length = max_length
        self.min_length = min_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, batch):
        index, ids = [], []
        for idx, instance in batch:
            instance = self.vocab.encode(instance, max_length=self.max_length)
            index.append(idx)
            ids.append(torch.LongTensor(instance))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        if torch.cuda.is_available():
            ids = ids.cuda()
        return index, ids

if __name__ == "__main__":
    args = vars(parser_args())
    
    # distribute
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    data = dblpDataset(
        read_file(f'data/{args["dataset"]}/{args["mode"]}/content_{args["mode"]}'),
        max_length=args['max_length'],
        min_length=args['min_length'],
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    dataloader = DataLoader(
        data, shuffle=False, sampler=train_sampler, batch_size=args['batch_size'], collate_fn=data.collate,
    )
    
    # apex speed up
    # model = Text2Tensor(fine_tune=args['fine_tune'], pool=args['pool'])
    # NOTE:
    model = BertForMultiLabelSequenceClassification(num_labels=99)
    model.cuda()
    # NOTE
    load_model(model, f'ckpt/{args["dataset"]}/best.pt')
    model = amp.initialize(model, optimizers=None, opt_level=args['opt_level'])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args['local_rank']],
        output_device=args['local_rank'],
    )
    
    index, bundle = [], []
    pbar = tqdm(dataloader)
    for batch in pbar:
        idx, ids = batch
        # embedding = model(ids).cpu().tolist()
        # NOTE:
        embedding = model(ids, test=True).cpu().tolist()
        
        for idx_, embed in zip(idx, embedding):
            bundle.append(embed)
            index.append(idx_)
        pbar.set_description(f'[!] dataset: {args["dataset"]}; mode: {args["mode"]}')
    
    with open(f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/{args["mode"]}_embd_{args["local_rank"]}.pkl', 'wb') as f:
        joblib.dump(bundle, f)
        
    with open(f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/{args["mode"]}_idx_{args["local_rank"]}.pkl', 'wb') as f:
        joblib.dump(index, f)
