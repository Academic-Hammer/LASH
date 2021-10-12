from dotmap import DotMap
import numpy as np
import scipy.io
import pickle
import os
from utils_cs import *
from tqdm import tqdm
import sklearn.preprocessing
from scipy import sparse
import argparse

##################################################################################################
# Parameters

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", default=32, help="0.5 * Number of bits of the embedded vector.", type=int) 
parser.add_argument("--train_batch_size", default=500, type=int)
parser.add_argument("--test_batch_size", default=200, type=int) 
parser.add_argument("--num_epochs", default=50, type=int) 

parser.add_argument("--lr", default=0.0003, type=float)
parser.add_argument("--gamma", default=0.00013, type=float)

parser.add_argument("--step_size", default=100, type=int, help="step size of lr drop.")
parser.add_argument("--drop", default=1, type=float, help="lr drop rate.")

parser.add_argument("--lam", default=1.0, type=float)
parser.add_argument("--beta", default=1, type=float) 

args = parser.parse_args()
print(args)

if not args.gpunum:
    parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the dataset.")

##################################################################################################
# Model

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter

class LASH(nn.Module):
    
    def __init__(self, vocabSize, latentDim, dropoutProb=0.):
        super(LASH, self).__init__()
        
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        
        self.dtype = torch.cuda.FloatTensor
        # document
        self.fc1_1 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.fc2_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_1 = nn.Linear(self.hidden_dim, self.latentDim)
        # network
        self.fc1_2 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.fc2_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_2 = nn.Linear(self.hidden_dim, self.latentDim)

        self.dropout = nn.Dropout(p=dropoutProb)
        
        self.relu = nn.LeakyReLU()

        self.fc4 = nn.Linear(self.latentDim * 2, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.vocabSize)
        
    def encode(self, document_mat, network_mat):
        
        h1_1 = self.relu(self.fc1_1(document_mat))
        h2_1 = self.relu(self.fc2_1(h1_1))
        h3_1 = self.dropout(h2_1)
        x1 = self.fc3_1(h3_1)

        h1_2 = self.relu(self.fc1_2(network_mat))
        h2_2 = self.relu(self.fc2_2(h1_2))
        h3_2 = self.dropout(h2_2)
        x2 = self.fc3_2(h3_2)

        # x = x1 + x2
        x = torch.cat((x1, x2), 1) 

        h = torch.sign(x)
        return x, h
    

    def decode(self, x):
        h4 = self.relu(self.fc4(x))
        y = self.fc5(h4)
        return y

    def forward(self, document_mat, network_mat):

        x, h = self.encode(document_mat, network_mat)
        y = self.decode(x)

        return y, x, h

##################################################################################################
# units

def compute_reconstr_loss(log_word_prob, document_mat): 
    loss = torch.norm(log_word_prob - document_mat, p=2, dim=1).sum()
    return loss / document_mat.shape[0]

def compute_hash_loss(x, s, k):
    s = Variable(torch.from_numpy(s).type(torch.cuda.FloatTensor))
    end_item = k * s
    start_item = torch.mm(x, x.t())
    return compute_reconstr_loss(end_item, start_item)

def transform(doc_mat, net_mat, batch_size, V): 
    model.eval()
    num_doc = doc_mat.shape[0]

    # pbar = tqdm(total=num_doc, ncols=0) 
    for idx in range(0, num_doc, batch_size):
        if idx + batch_size < doc_mat.shape[0]:
            batch_train = doc_mat[idx:idx+batch_size]
            batch_n_train = net_mat[idx:idx+batch_size]
        else:
            batch_train = doc_mat[idx:]
            batch_n_train = net_mat[idx:]

        batch_train = Variable(torch.from_numpy(batch_train).type(torch.cuda.FloatTensor))
        batch_n_train = Variable(torch.from_numpy(batch_n_train).type(torch.cuda.FloatTensor))
        X, _ = model.encode(batch_train, batch_n_train)

        V.extend(list(X.cpu().data.numpy()))

    #     pbar.set_description("transform iteration {}".format(idx))
    #     pbar.update(len(batch_train))
    # pbar.close()

def sign_transform(X):
    binary_code = np.zeros(X.shape)
    for i in range(X.shape[1]):
        binary_code[np.nonzero(X[:,i] < 0),i] = -1
        binary_code[np.nonzero(X[:,i] >= 0),i] = 1
    return binary_code.astype(int)

def run_validation(refer_embeddings, refer_n_embeddings, refer_categories, query_embeddings, query_n_embeddings, query_categories):
    model.eval()

    # embeddings -》 binary code
    batch_size = args.test_batch_size
    refer_embeddings = np.array(refer_embeddings)
    refer_n_embeddings = np.array(refer_n_embeddings)
    # print('reference database: ', refer_embeddings.shape)
    query_embeddings = np.array(query_embeddings)
    query_n_embeddings = np.array(query_n_embeddings)
    # print('query database: ', query_embeddings.shape)

    ## transform reference
    v_references = []
    transform(refer_embeddings, refer_n_embeddings, batch_size, v_references) 
    v_references = np.array(v_references)
    # print('v reference database: ', v_references.shape)
    ## transform query
    v_queries = []
    transform(query_embeddings, query_n_embeddings, batch_size, v_queries)
    v_queries = np.array(v_queries)
    # print('v query database: ', v_queries.shape)
    # print("1. forward finished !")
    
    b_references = sign_transform(v_references)
    # print(v_references[0])
    # print(b_references[0])
    b_queries = sign_transform(v_queries)
    # print("2. binary finished !")
    return run_topK_retrieval_experiment_GPU_batch_train(b_references, b_queries, 
                                  refer_categories, query_categories, batch_size, TopK=100)

##################################################################################################
# Load Data

DATASET = args.dataset
train_embeddings, train_n_embeddings, train_categories = Load_Dataset("./data/dblp_{}/train/".format(DATASET), 'train') 
print("1. load train date finished !")
validation_embeddings, validation_n_embeddings, validation_categories = Load_Dataset("./data/dblp_{}/validation/".format(DATASET), 'validation')
print("2. load validation date finished !")
test_embeddings, test_n_embeddings, test_categories = Load_Dataset("./data/dblp_{}/test/".format(DATASET), 'test')
print("3. load test date finished !")

num_trains = len(train_embeddings)
print('num trains:{}'.format(num_trains))
num_validation = len(validation_embeddings)
print('num validation:{}'.format(num_validation))
num_test = len(test_embeddings)
print('num test:{}'.format(num_test))


transform_gnd(num_trains, train_categories, DATASET)
train_categories = np.array(train_categories)
# print(train_categories.shape)
# print("transform train_categories finished !")

transform_gnd(num_validation, validation_categories, DATASET)
validation_categories = np.array(validation_categories)
# print(validation_categories.shape)
# print("transform validation_categories finished !")

transform_gnd(num_test, test_categories, DATASET)
test_categories = np.array(test_categories)
# print(test_categories.shape)
# print("transform test_categories finished !")

##################################################################################################
# Train and Validation

GPU_NUM = args.gpunum
NUM_BITS = args.nbits
bits_ture = 2 * NUM_BITS 

num_feas = len(train_embeddings[0])

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

model = LASH(num_feas, NUM_BITS, dropoutProb=0.1)
print(model)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma = args.drop)

BATCH_SIZE = args.train_batch_size
NUM_EPOCHS = args.num_epochs

quanWeight = 0. 
quanStepSize = 1 / (100*num_trains)

hashWeight = args.beta

BestPrec = 0.
BestRound = 0

for iteration in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = []
    lr_temp = optimizer.state_dict()['param_groups'][0]['lr']

    pbar = tqdm(total=num_trains, ncols=0) 
    for idx in range(0, num_trains, BATCH_SIZE): 
        if idx + BATCH_SIZE < num_trains:
            batch_train = train_embeddings[idx:idx+BATCH_SIZE]
            batch_n_train = train_n_embeddings[idx:idx+BATCH_SIZE]
            batch_categories = train_categories[idx:idx+BATCH_SIZE]
        else:
            batch_train = train_embeddings[idx:]
            batch_n_train = train_n_embeddings[idx:]
            batch_categories = train_categories[idx:]

        batch_train = np.array(batch_train)
        batch_n_train = np.array(batch_n_train)
        # print(batch_train.shape) 

        batch_train = Variable(torch.from_numpy(batch_train).type(torch.cuda.FloatTensor))
        batch_n_train = Variable(torch.from_numpy(batch_n_train).type(torch.cuda.FloatTensor))
        optimizer.zero_grad()
        
        # print(batch_train.shape) 

        y, x, h = model(batch_train, batch_n_train) # forward
        # print(h[0])
        reconstr_loss = compute_reconstr_loss(y, batch_train)
        reconstr_loss_n = compute_reconstr_loss(y, batch_n_train)
        quan_loss = compute_reconstr_loss(x,h)
        
        s = compute_similarity(batch_categories, batch_categories, args.lam) 
        hash_loss = compute_hash_loss(x, s, bits_ture)
        
        loss = reconstr_loss + reconstr_loss_n + (quanWeight * quan_loss) + (hashWeight * hash_loss)

        loss.backward()
        optimizer.step()

        quanWeight = min(quanWeight + quanStepSize, args.gamma)
        
        train_loss.append(loss.item())

        pbar.set_description("{}: LASH Best Round:{} WNDCG:{:.4f} AvgLoss:{:.3f} quanWeight:{:.6f} "
                             .format(iteration, BestRound, BestPrec, np.mean(train_loss), quanWeight))
        pbar.update(len(batch_train))
    
    scheduler.step()
    pbar.close()
    
    # validation in train
    # if (iteration-1) % 5 == 0:
    # prec, ndcg, wndcg = run_validation(train_embeddings, train_n_embeddings, train_categories, validation_embeddings, validation_n_embeddings, validation_categories)
    prec, ndcg, wndcg = run_validation(train_embeddings, train_n_embeddings, train_categories, test_embeddings, test_n_embeddings, test_categories)
    print("prec in this epoch: ", prec)
    print("ndcg in this epoch: ", ndcg)
    print("wndcg in this epoch: ", wndcg)

    BestPrec = max(BestPrec, wndcg)
    
    if BestPrec == wndcg:
        BestRound = iteration
