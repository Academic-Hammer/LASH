import numpy as np
import os
import scipy.io
from dotmap import DotMap
from tqdm import tqdm
import json
import pickle
import torch
from scipy.sparse import coo_matrix
import math
import joblib

################################################################################################################
def Load_Dataset(filename, fold):
    # embeddings_path = filename+"bert_embedding/embeddings"
    # embeddings_n_path = filename + "network_embedding/embeddings"
    # embeddings = pickle.load(open(embeddings_path, 'rb'), encoding='utf-8')
    # embeddings_n = pickle.load(open(embeddings_n_path, 'rb'), encoding='utf-8')

    embeddings_path = filename+"bert_embedding/embeddings_new"
    embeddings_n_path = filename + "network_embedding/embeddings"
    embeddings = pickle.load(open(embeddings_path, 'rb'), encoding='utf-8')
    embeddings_n = pickle.load(open(embeddings_n_path, 'rb'), encoding='utf-8')

    indeces_path = filename+"bert_embedding/indeces" 
    indeces_n_path = filename + "network_embedding/indeces"

    categories_path = filename + "category_" + fold + "_norm"

    # indeces = pickle.load(open(indeces_path, 'rb'), encoding='utf-8')
    # indeces_n = pickle.load(open(indeces_n_path, 'rb'), encoding='utf-8')
  
    categories = []
    with open(categories_path, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        pbar = tqdm(total=num_lines, ncols=0) 
        for iteration in range(num_lines):
            line = lines[iteration]
            line = line.split("\t", 1)
            categories.append(json.loads(line[1]))
            pbar.set_description("categories list iteration {}".format(iteration))
            pbar.update(1)
        pbar.close()
    return embeddings, embeddings_n, categories

################################################################################################################

def transform_gnd(num, gnd, DATASET):
    # trun sparse to dense
    num_cate = {'ml':99, 'cv':97, 'cn':98, 'nlp':86}
    # pbar = tqdm(total=num, ncols=0) 
    for iteration in range(num):
        cate = gnd[iteration]
        data = cate['data']
        col = cate['col']
        row = [0 for i in range(len(data))]
        dense = np.squeeze(np.array(coo_matrix((data,(row,col)),shape=(1,num_cate[DATASET])).todense()))
        gnd[iteration] = dense
        # pbar.set_description("transform gnd iteration {}".format(iteration))
        # pbar.update(1)
    # pbar.close()


def compute_similarity(test_categories, train_categories, lam): 

    n_test = test_categories.shape[0]
    n_train = train_categories.shape[0]

    # compute jaccard
    test_categories_bin = np.sign(test_categories)
    train_categories_bin = np.sign(train_categories)

    test_and_train = test_categories_bin.dot(train_categories_bin.T) 
    
    test_sum = test_categories_bin.sum(1)
    test_array = test_sum[:, np.newaxis].repeat(n_train, axis=1)
    train_sum = train_categories_bin.sum(1)
    train_array = train_sum[np.newaxis, :].repeat(n_test, axis=0)

    test_or_train = test_array + train_array - test_and_train

    Jaccard = test_and_train / test_or_train

    # print("1. Compute Jaccard finished !")
    
    norm_test = np.linalg.norm(test_categories, axis=-1).reshape(n_test,1)
    norm_train = np.linalg.norm(train_categories,axis=-1).reshape(1,n_train)
    end_norm = np.dot(norm_test,norm_train)
    Cos = np.dot(test_categories, train_categories.T)/end_norm

    # print("2. Compute Cos finished !")

    return ( Jaccard + lam * Cos ) / (1 + lam)

################################################################################################################


def micro_prec(query_TopK_indeces, gnd_train, gnd_test, TopK):
    
    n_test = len(query_TopK_indeces)
    print(n_test)

    gnd_train = np.sign(gnd_train) 
    gnd_test = np.sign(gnd_test)

    prec = []
    # pbar = tqdm(total=n_test, ncols=0)
    for i in range(n_test):
        received_cate = gnd_test[i]
        # print(received_cate.shape)
        # print(received_cate)

        total_cate = gnd_train[list(query_TopK_indeces[i])].sum(0)
        # print(total_cate.shape)
        # print(total_cate)

        tp = total_cate[list(np.nonzero(received_cate)[0])].sum()
        tp_and_fp = total_cate.sum()
        prec.append(tp/tp_and_fp)
    #     pbar.set_description("prec iteration {}".format(i))
    #     pbar.update(1)
    # pbar.close()

    return sum(prec)/len(prec)


def NDCG_gpu(query_TopK_indeces, gnd_train, gnd_test, TopK, weighted=False):
    n_test = len(query_TopK_indeces)
    # print(n_test)

    if not weighted:
        gnd_train = np.sign(gnd_train) 
        gnd_test = np.sign(gnd_test) 

    weight = np.array([math.log(i+1, 2) for i in range(1,TopK+1)])
    weight = weight[::-1].copy()

    gnd_train = torch.cuda.FloatTensor(gnd_train)
    weight = torch.cuda.FloatTensor(weight)
    gnd_test1 = torch.cuda.FloatTensor(gnd_test)

    NDCG = []
    pbar = tqdm(total=n_test, ncols=0)
    for i in range(n_test):
        received_cate = gnd_test[i]
        
        gnd_train1 = torch.min(gnd_test1[i], gnd_train) 
        
        # print(received_cate)
        received_cate = list(np.nonzero(received_cate)[0])
        # print(received_cate)

        gnd_train_this_test = gnd_train1[:,received_cate].sum(1) 

        gnd_TopK = gnd_train_this_test[list(query_TopK_indeces[i])]
        # print(gnd_TopK)
        gnd_train_this_test = gnd_train_this_test.sort()[0]
        gnd_bestK = gnd_train_this_test[-TopK:]
        # print(gnd_bestK)

        DCG = torch.div(gnd_TopK, weight).sum()
        # print(DCG)
        IDCG = torch.div(gnd_bestK, weight).sum()

        NDCG.append(float((DCG/IDCG).cpu().data))
        pbar.set_description("ndcg iteration {}".format(i))
        pbar.update(1)
    pbar.close()

    return sum(NDCG)/len(NDCG)

################################################################################################################

def run_topK_retrieval_experiment_GPU_batch_train(codeTrain, codeTest, 
                                                  gnd_train, gnd_test, batchSize=200, TopK=100):
    
    n_bits = codeTrain.shape[1]
    n_train = codeTrain.shape[0]
    n_test = codeTest.shape[0]

    #from tqdm import tqdm_notebook as tqdm
    assert (codeTrain.shape[1] == codeTest.shape[1])
    assert (gnd_train.shape[1] == gnd_test.shape[1])
    assert (codeTrain.shape[0] == gnd_train.shape[0])
    assert (codeTest.shape[0] == gnd_test.shape[0])

    query_TopK_indeces = []
    codeTrain = torch.cuda.FloatTensor(codeTrain.T)
    # print(codeTrain.shape)
    for batchIdx in tqdm(range(0, n_test, batchSize), ncols=0):
        s_idx = batchIdx
        e_idx = min(batchIdx + batchSize, n_test)
        numQuery = e_idx - s_idx

        batch_codeTest = codeTest[s_idx:e_idx]
        batch_codeTest = torch.cuda.FloatTensor(batch_codeTest)

        scores = torch.mm(batch_codeTest, codeTrain)
        # print(scores.shape)
        
        scores = scores.sort()
        indeces_TopK = scores[1][:, -TopK:] 
        
        query_TopK_indeces.extend(list(indeces_TopK.cpu().data.numpy())) 

    # Evaluation
    # return micro_prec(query_TopK_indeces, gnd_train, gnd_test, TopK)
    # return NDCG_gpu(query_TopK_indeces, gnd_train, gnd_test, TopK)
    # return 0,0,NDCG_gpu(query_TopK_indeces, gnd_train, gnd_test, TopK, True)
    return micro_prec(query_TopK_indeces, gnd_train, gnd_test, TopK), NDCG_gpu(query_TopK_indeces, gnd_train, gnd_test, TopK), NDCG_gpu(query_TopK_indeces, gnd_train, gnd_test, TopK, True)
