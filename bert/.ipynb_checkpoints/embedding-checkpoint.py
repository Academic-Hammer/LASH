from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import ipdb, gensim, nltk
import numpy as np
from tqdm import tqdm

class BertForMultiLabelSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, num_labels=2, dropout=0.2):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, test=False):
        if test:
            with torch.no_grad():
                _, pooled_output = self.bert(input_ids)
                return pooled_output    # [B, 768]
        _, pooled_output = self.bert(input_ids)    # [B, H]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)    # [B, N]

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), 
                labels.view(-1, self.num_labels),
            )
            return loss
        else:
            return logits

class Text2Tensor(nn.Module):

    def __init__(self, fine_tune=False, pool=False):
        super(Text2Tensor, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.fine_tune = fine_tune
        self.pool = pool

    def forward(self, ids):
        '''convert ids to embedding tensor
        Return: [B, 768]
        '''
        if self.fine_tune:
            embd = self.model(ids)[0]    # [B, S, 768]
        else:
            with torch.no_grad():
                embd = self.model(ids)[0]
        if self.pool:
            rest = torch.mean(embd, dim=1)    # [B, 768]
        else:
            rest = embd[:, 0, :]
        return rest
    
class GloVe2Tensor:
    
    def __init__(self):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('data/english_w2v.bin', binary=True)
        print(f'[!] load english word2vec by gensim; GoogleNews WordVector: data/english_w2v.bin')
        
    def convert(self, texts):
        rest = []
        for text in tqdm(texts):
            words = nltk.word_tokenize(text)
            vectors = []
            for w in words:
                if w in self.w2v:
                    vectors.append(self.w2v[w])
            if not vectors:
                vectors.append(np.random.randn(300))
            vectors = np.stack(vectors).mean(axis=0)
            rest.append(vectors)
        return rest    # [B, 300]

if __name__ == "__main__":
    pass
