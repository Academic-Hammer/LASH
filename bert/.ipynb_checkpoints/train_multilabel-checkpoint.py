from .embedding import BertForMultiLabelSequenceClassification
import argparse
import ipdb
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import transformers
from torch.nn.utils.rnn import pad_sequence
from apex import amp

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--min_length', default=50, type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--opt_level', default='O2', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    return parser.parse_args()

def read_file(path_data, path_category):
    # read data 
    with open(path_data) as f:
        data = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            index, content = line.split('\t', 1)
            data.append((index, content))
    # read category
    with open(path_category) as f:
        category = []
        for line in tqdm(f.readlines()):
            index, content = line.strip().split('\t', 1)
            content = eval(content)
            col = content["col"]
            category.append((index, col))
    assert len(data) == len(category)
    dataset = [(index, content, col) for (index, content), (index_, col) in zip(data, category)]
    return dataset

class dblpMLDataset(Dataset):

    def __init__(self, data, max_length=256, min_length=50):
        super(dblpMLDataset, self).__init__()
        self.vocab = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = data
        self.max_length = max_length
        self.min_length = min_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, batch):
        index, ids, label = [], [], []
        for idx, instance, label_ in batch:
            instance = self.vocab.encode(instance, max_length=self.max_length)
            index.append(idx)
            p = [0.] * 99
            for i in label_:
                p[i] = 1.
            label.append(p)
            ids.append(torch.LongTensor(instance))
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
        label = torch.tensor(label)
        if torch.cuda.is_available():
            ids = ids.cuda()
            label = label.cuda()
        return index, ids, label

if __name__ == "__main__":
    args = vars(parser_args())
    
    # distribute
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    data = dblpMLDataset(
        read_file(
            f'data/{args["dataset"]}/{args["mode"]}/content_{args["mode"]}',
            f'data/{args["dataset"]}/{args["mode"]}/category_{args["mode"]}_norm',
        ),
        max_length=args['max_length'],
        min_length=args['min_length'],
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    dataloader = DataLoader(
        data, shuffle=False, 
        sampler=train_sampler, 
        batch_size=args['batch_size'], 
        collate_fn=data.collate,
    )
    
    # apex speed up
    model = BertForMultiLabelSequenceClassification(num_labels=99)
    model.cuda()
    optimizer = transformers.AdamW(
        model.parameters(), 
        lr=args['lr'],
    )
    model, optimizer = amp.initialize(
        model, 
        optimizer, 
        opt_level=args['opt_level'],
    )
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args['local_rank']],
        output_device=args['local_rank'],
    )
    
    index, bundle = [], []
    pbar = tqdm(dataloader)
    for _ in tqdm(range(args['epoch'])):
        batch_num, total_loss = 0, 0
        for batch in pbar:
            optimizer.zero_grad()
            idx, ids, label = batch
            loss = model(ids, labels=label)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(optimizer), args['grad_clip'])
            optimizer.step()
            loss = loss.item()
            batch_num += 1
            total_loss += loss
            pbar.set_description(f'[!] loss: {round(loss, 4)}|{round(total_loss/batch_num, 4)}')
        if args['local_rank'] == 0:
            torch.save(model.state_dict(), f'ckpt/{args["dataset"]}/best.pt')
    print(f'[!] train {args["dataset"]} done')
            
