import os
import pickle
from collections import Counter
import dgl
import torch
from torch.utils.data import Dataset

import numpy as np
from dgl.nn.pytorch import EdgeWeightNorm
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
import json
import random
import math
import re
from collections import OrderedDict, Counter
from sklearn.metrics.pairwise import cosine_similarity

from utils.config_knn import *

from transformers import BertModel, BertTokenizer


params = config()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(params.seed)

def write_dict2json(dic, json_file):
    # Convert and write JSON object to file
    with open(json_file, "w") as outfile: 
        json.dump(dic, outfile)
    # print("Writing dict to json successfully")
    
def filters(sequence_Example):
    return re.sub(r"[UZOB]", "X", sequence_Example)

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_kmers_embs(file):
    with open(file, 'rb') as f:
        embed_matrix = pickle.load(f)
    # print('Kmers embeddings load successfully')
    return embed_matrix

def negative_pool_gen(data):
    pos_pairs, neg_pool = {}, {}
    for vir, hos in zip(data[0], data[1]):  
        if vir not in pos_pairs:
            pos_pairs[vir] = []
        pos_pairs[vir].append(hos)
    for vir, hos in pos_pairs.items():
        cand = list(set(data[1]) - set(hos))
        neg_pool[vir] = cand
    return neg_pool

def process_df(df):
    df['virus_seq'] = df['virus_seq'].apply(lambda x: filters(x))
    df['host_seq'] = df['host_seq'].apply(lambda x: filters(x))
    virus, host = df['virus_seq'].tolist(), df['host_seq'].tolist()
    lbls = [1 for _ in range(len(virus))]
    assert len(virus) == len(host), 'During df process, num. of virus != host'
    pos_data = (virus, host, lbls)
    return pos_data

def data_statics(df):
    pos_data = process_df(df)
    neg_pool = negative_pool_gen(pos_data)
    max_len_virus = df['virus_seq'].str.len().max()
    max_len_host = df['host_seq'].str.len().max()
    if max_len_virus > max_len_host:
        max_len = max_len_virus
    else:
        max_len = max_len_host
    print(f'Num. of data in the original file is: {len(df)}, and the max length is: {max_len}')
    return neg_pool, max_len

def data_read(data_path, end_str='new.csv'):
    for file in os.listdir(data_path):
        if file.endswith(end_str):
            file_name = data_path+file
            df = pd.read_csv(file_name)
            # df = df[:500]
            neg_pool, max_len = data_statics(df)
            pos_data = process_df(df)
    return pos_data, neg_pool, max_len


##########update1#########
# def prepro_data(data_path, neg_type, end_str='new.csv', kmers=5, emb_file='./checkpoints/Pair_feature/', kmer_file='./Dataset/Orthomy/'):
    # emb_file += f'bert_k{kmers}_d1024_embs.pkl'
    # kmer_file += f'kmers2dict_k{kmers}.json'
##########update1#########
def prepro_data(data_path, neg_type, data_name, kmers, end_str='new.csv', emb_file='./checkpoints/Pair_feature/', kmer_file='./Dataset/'):
    emb_file += f'{data_name}_bert_k{kmers}_d1024_embs.pkl'
    kmer_file += f'{data_name}/kmers2dict_k{kmers}.json'
    pos_data, neg_pool, max_len = data_read(data_path, end_str=end_str)
    kmers_embs = load_kmers_embs(emb_file)
    kmers_dict = read_json(kmer_file)
    

    def get_neg_data(pos_data, neg_pool, neg_type):#  seq_sim
        print(f'The type of generating negative data is {neg_type}')
        neg_data = []
        virus, neg_hos = [], []
        for vir, hos in zip(pos_data[0], pos_data[1]):
            all_negs = neg_pool[vir]
            negs_sel = random.sample(all_negs, int(np.ceil(len(all_negs)/5)))
            if neg_type == 'random':
                neg_hos.append(random.sample(negs_sel, 1)[0])
                virus.append(vir)
        neg_lbls = [0 for _ in range(len(virus))]
        print(f'Num of pos data is {len(virus)}, neg data is {len(neg_hos)}')
        assert len(virus) == len(neg_hos), 'In train-test data, num. of pos data != neg data'
        neg_data = (virus, neg_hos, neg_lbls)
        return neg_data
    neg_data = get_neg_data(pos_data, neg_pool, neg_type) 
    
    def data2kmers(pos_data, neg_data, kmers):
        assert len(pos_data[0]) == len(neg_data[0]) & len(pos_data[1]) == len(neg_data[1]), 'In data2kmers, num. of virus data or host data in kmers not EQUAL!'
        pos_kmers = ([[vir[i:i + kmers] for i in range(len(vir)-kmers+1)] for vir in pos_data[0]], \
        [[hos[i:i + kmers] for i in range(len(hos)-kmers+1)] for hos in pos_data[1]], pos_data[2])
        neg_kmers = ([[vir[i:i + kmers] for i in range(len(vir)-kmers+1)] for vir in neg_data[0]], \
        [[hos[i:i + kmers] for i in range(len(hos)-kmers+1)] for hos in neg_data[1]], neg_data[2])
        #merge pos and neg data
        kmer_data = (pos_kmers[0]+neg_kmers[0], pos_kmers[1]+neg_kmers[1], pos_kmers[2]+neg_kmers[2])
        return kmer_data
    
    def kmer2id(k_virus, k_host):
        idSeq_virus = []
        for virus in k_virus:
            temp_seq = []
            for vir in virus:
                temp_seq.append(kmers_dict[vir])
            idSeq_virus.append(temp_seq)
        
        idSeq_host = []
        for host in k_host:
            temp_seq = []
            for hos in host:
                temp_seq.append(kmers_dict[hos])
            idSeq_host.append(temp_seq)
        return idSeq_virus, idSeq_host
                    
    def seqs_embs(pos_data, neg_data, kmers):
        pair_embs = []
        kmer_data = data2kmers(pos_data, neg_data, kmers)
        idSeq_virus, idSeq_host = kmer2id(kmer_data[0], kmer_data[1])
        lbls = kmer_data[2]
        print('same len', len(idSeq_virus)==len(idSeq_host))
        for id_vir, id_hos in zip(idSeq_virus, idSeq_host):
            vir_emb = np.mean([kmers_embs[kmer] for kmer in id_vir], axis=0) 
            hos_emb = np.mean([kmers_embs[kmer] for kmer in id_hos], axis=0)
            pair_emb = np.concatenate((vir_emb, hos_emb), axis=0)
            # print(vir_emb.shape, hos_emb.shape, pair_emb.shape)
            pair_embs.append(pair_emb)
        return pair_embs, lbls
    
    pair_embs, lbls = seqs_embs(pos_data, neg_data, kmers)    
    return pair_embs, lbls
    
# def main():
#     raw_dir=f'./Dataset/{params.data_name}/'
#     neg_type = params.neg_type
#     is_train = False
#     seed = params.seed
#     kmers = params.k
#     kmer_data, _, max_len = prepro_data(raw_dir, neg_type, is_train=is_train, end_str='new_test.csv', kmers=kmers)
    # print('max_len:', max_len, '\nkmer_data:', kmer_data)

class ProteinDataset(Dataset):
    def __init__(self, raw_dir, save_dir):
        """
        Args:
            raw_dir (str): Path to the raw data directory.
            neg_type (str): The type of negative samples.
            data_name (str): The name of the dataset.
            file_path (str): Path to save or load the dataset.
        """
        self.file_path = save_dir + params.data_name + '_' + 'k'+str(params.k)+'_'+params.neg_type+ '_cf'+ str(params.fold)+".bin"

        # 检查文件是否存在
        if os.path.exists(self.file_path):
            print(f"File {self.file_path} exists. Loading dataset...")
            self.load(self.file_path)
        else:
            print(f"File {self.file_path} does not exist. Preprocessing and saving dataset...")
            self.pair_embs, self.labels =prepro_data(raw_dir, params.neg_type, params.data_name, params.k, end_str='new.csv')
            # 保存数据
            self.save(self.file_path)

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sequence and label to retrieve.

        Returns:
            tuple: (seqs_virus, seqs_host, label), where sequences are tensors and label is the corresponding label.
        """
        pair_emb = self.pair_embs[idx]
        label = self.labels[idx]
        return pair_emb, label

    def save(self, file_path):
        """Save the dataset to disk."""
        torch.save({
            'pair_embs': self.pair_embs,
            'labels': self.labels
        }, file_path)
        print(f"Dataset saved to {file_path}")

    def load(self, file_path):
        """Load the dataset from disk."""
        data = torch.load(file_path)
        self.pair_embs = data['pair_embs']
        self.labels = data['labels']
        print(f"Dataset loaded from {file_path}")

