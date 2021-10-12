import joblib
import argparse
from tqdm import tqdm

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--mode', default='train', type=str)
    return parser.parse_args()

def read_index(path):
    with open(path) as f:
        data = []
        for line in tqdm(f.readlines()):
            line = line.strip()
            index = line.split('\t', 1)[0]
            data.append(index)
    return data

def read_file(path):
    with open(path, 'rb') as f:
        data = joblib.load(f)
    return data

def construct(dataset, mode, original_index):
    dataset, dataset_dict = [], {}
    # make sure torch.distributed.launch only create 2 processes
    for i in tqdm([0, 1]):
        path_1 = f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/{args["mode"]}_embd_{i}.pkl'
        path_2 = f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/{args["mode"]}_idx_{i}.pkl'
        data, index = read_file(path_1), read_file(path_2)
        dataset.extend(data)
        for idx, idx_ in enumerate(index):
            dataset_dict[idx_] = idx
    dataset = [dataset[dataset_dict[i]] for i in original_index]
    print(f'[!] reconstruct the dataset dict over')
    
    with open(f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/embeddings_new', 'wb') as f:
        joblib.dump(dataset, f)

if __name__ == "__main__":
    args = vars(parser_args())
    original_index = read_index(f'data/{args["dataset"]}/{args["mode"]}/content_{args["mode"]}')
    with open(f'data/{args["dataset"]}/{args["mode"]}/bert_embedding/indices_new', 'wb') as f:
        joblib.dump(original_index, f)
    construct(args['dataset'], args['mode'], original_index)