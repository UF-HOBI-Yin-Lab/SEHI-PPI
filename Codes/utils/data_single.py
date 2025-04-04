import os
import pickle
from collections import Counter
import dgl
from dgl.data import DGLDataset, save_graphs, load_graphs
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

from utils.config_single import *

params = config()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(params.seed)

def gpu_check():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU Device {i}: {device_name}")

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

def save_kmers_embs(kmers_embs, file):
    with open(file, 'wb') as f:
        pickle.dump(kmers_embs, f)
    # print('Kmers embeddings save successfully')
    
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
            # df = df[:300]
            neg_pool, max_len = data_statics(df)
            pos_data = process_df(df)
    return pos_data, neg_pool, max_len

def kmer_mapping(seqs, kmers=5):
    kmers_dict = {}
    idx = 1
    for seq in seqs:
        tmp = [seq[i:i + kmers] for i in range(len(seq)-kmers+1)]
        for kmer in tmp:
            if kmer not in kmers_dict:
                kmers_dict[kmer] = idx
                idx += 1
    return kmers_dict

def prepro_data(data_path, neg_type, data_name, end_str='new.csv', kmers=5, emb_file='./checkpoints/Pair_feature/', kmer_file='./Dataset/'):
    emb_file += f'{data_name}_bert_k{kmers}_d1024_embs.pkl'
    kmer_file += f'{data_name}/kmers2dict_k{kmers}.json'
    
    pos_data, neg_pool, max_len = data_read(data_path, end_str=end_str)
    kmers_embs = load_kmers_embs(emb_file)
    kmers_dict = read_json(kmer_file)
    
    def position_generator(max_num, lamada=1.0):
        rand = 1 - np.random.uniform(0, 1)
        pos_id = - math.log(rand)/lamada
        return min(max_num-1, int(pos_id))

    def insert_descending(d, key, value):
        d[key] = value
        sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=False)
        return OrderedDict(sorted_items)

    def dyn_insert_dict(key, value, numbers_dict): 
        numbers_dict = insert_descending(numbers_dict, key, value)
        return numbers_dict
       
    def data2kmers(pos_data, neg_data, kmers):
        assert len(pos_data[0]) == len(neg_data[0]) & len(pos_data[1]) == len(neg_data[1]), 'In data2kmers, num. of virus data or host data in kmers not EQUAL!'
        pos_kmers = ([[vir[i:i + kmers] for i in range(len(vir)-kmers+1)] for vir in pos_data[0]], \
        [[hos[i:i + kmers] for i in range(len(hos)-kmers+1)] for hos in pos_data[1]], pos_data[2])
        neg_kmers = ([[vir[i:i + kmers] for i in range(len(vir)-kmers+1)] for vir in neg_data[0]], \
        [[hos[i:i + kmers] for i in range(len(hos)-kmers+1)] for hos in neg_data[1]], neg_data[2])
        #merge pos and neg data
        kmer_data = (pos_kmers[0]+neg_kmers[0], pos_kmers[1]+neg_kmers[1], pos_kmers[2]+neg_kmers[2])
        return kmer_data

    def count_common_elements(list1, list2):
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        # Find the intersection of the two Counters
        common_elements = counter1 & counter2
        # Sum the counts of the common elements
        common_count = sum(common_elements.values())
        return common_count

    def get_neg_data(pos_data, neg_pool, kmers_embs, neg_type, kmers=5):#  seq_sim
        print(f'The type of generating negative data is {neg_type}')
        neg_data = []
        virus, neg_hos = [], []
        for vir, hos in zip(pos_data[0], pos_data[1]):
            all_negs = neg_pool[vir]
            negs_sel = random.sample(all_negs, int(np.ceil(len(all_negs)/5)))
            if neg_type == 'random':
                neg_hos.append(random.sample(negs_sel, 1)[0])
                virus.append(vir)
            elif neg_type == 'kmers_stat':
                hos_kmers = [hos[i:i + kmers] for i in range(len(hos)-kmers+1)]
                sel_negs_kmers_dict = {neg: [neg[i:i + kmers] for i in range(len(neg)-kmers+1)] for neg in negs_sel}
                nb_match = float("inf")
                best_neg = None
                for neg, nkmers in sel_negs_kmers_dict.items():
                    nb = count_common_elements(hos_kmers, nkmers)#len(set(hos_kmers).intersection(set(nkmers)))
                    if nb < nb_match:
                        best_neg = neg
                        nb_match = nb
                virus.append(vir)
                neg_hos.append(best_neg) 
            elif neg_type == 'seq_sim':
                # print(f'virus: {vir}, host: {hos}, negs_sel: {negs_sel}')
                hos_kmers = [hos[i:i + kmers] for i in range(len(hos)-kmers+1)]
                hos_kmers_emb = np.mean([kmers_embs[kmers_dict[kmer]] for kmer in hos_kmers], axis=0)
                # hos_kmers_emb = np.sum([kmers_embs[kmers_dict[kmer]] for kmer in hos_kmers], axis=0)
                sel_negs_kmers_dict = {neg: [neg[i:i + kmers] for i in range(len(neg)-kmers+1)] for neg in negs_sel}
                sim_score = float("inf")
                best_neg = None
                for neg, nkmers in sel_negs_kmers_dict.items():
                    nkmers_emb = np.mean([kmers_embs[kmers_dict[kmer]] for kmer in nkmers], axis=0)
                    # nkmers_emb = np.sum([kmers_embs[kmers_dict[kmer]] for kmer in nkmers], axis=0)
                    sim = cosine_similarity([hos_kmers_emb], [nkmers_emb])[0][0]
                    if sim < sim_score:
                        best_neg = neg
                        sim_score = sim
                    # print(f'hos_kmers: {hos_kmers}, nkmers: {nkmers}, sim: {sim}, best_neg: {best_neg}')
                virus.append(vir)
                neg_hos.append(best_neg)
            elif neg_type == 'adap_sam':
                # print(f'virus: {vir}, host: {hos}, negs_sel: {negs_sel}')
                hos_kmers = [hos[i:i + kmers] for i in range(len(hos)-kmers+1)]
                hos_kmers_emb = np.mean([kmers_embs[kmers_dict[kmer]] for kmer in hos_kmers], axis=0)
                # hos_kmers_emb = np.sum([kmers_embs[kmers_dict[kmer]] for kmer in hos_kmers], axis=0)
                sel_negs_kmers_dict = {neg: [neg[i:i + kmers] for i in range(len(neg)-kmers+1)] for neg in negs_sel}
                order_dict = OrderedDict()
                for neg, nkmers in sel_negs_kmers_dict.items():
                    nkmers_emb = np.mean([kmers_embs[kmers_dict[kmer]] for kmer in nkmers], axis=0) #make a big diff
                    # nkmers_emb = np.sum([kmers_embs[kmers_dict[kmer]] for kmer in nkmers], axis=0)
                    sim = cosine_similarity([hos_kmers_emb], [nkmers_emb])[0][0]
                    order_dict = dyn_insert_dict(neg, sim, order_dict)
                    # print(f'hos_kmers: {hos_kmers}, nkmers: {nkmers}, sim: {sim}, dict: {order_dict}')
                neg_id = position_generator(len(negs_sel))
                best_neg = list(order_dict.keys())[neg_id]
                # print(f'neg_id: {neg_id}, best_neg: {best_neg}')
                virus.append(vir)
                neg_hos.append(best_neg)
        neg_lbls = [0 for _ in range(len(virus))]
        print(f'Num of pos data is {len(virus)}, neg data is {len(neg_hos)}')
        assert len(virus) == len(neg_hos), 'In train-test data, num. of pos data != neg data'
        neg_data = (virus, neg_hos, neg_lbls)
        return neg_data
    
    neg_data = get_neg_data(pos_data, neg_pool, kmers_embs, neg_type, kmers) 
    kmer_data = data2kmers(pos_data, neg_data, kmers)    
    return kmer_data, kmers_dict, max_len
        
class ProteinDataset(DGLDataset):
    """
        raw_dir : str
            Specifies the directory where the downloaded data is stored or where the downloaded data is stored. Default: ~/.dgl/
        save_dir : str
            The directory where the finished dataset will be saved. Default: the value specified by raw_dir
        force_reload : bool
            If or not to re-import the dataset. Default: False
        verbose : bool
            Whether to print progress information.
        """

    def __init__(self,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(ProteinDataset, self).__init__(name='protein',
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose
                                            )
        # print('Dataset initialization is completed!\n')
        
    def process(self):
        # Processing of raw data into plots, labels
        self.kmers = params.k  # 这里k是4
        self.neg_type = params.neg_type

        kmer_data, self.kmers_dict, self.max_seq_len = prepro_data(self.raw_dir, self.neg_type, params.data_name, end_str='new.csv', kmers=self.kmers)
        # # Get labels and k-mer sentences
        self.k_virus, self.k_host, self.labels = kmer_data[0], kmer_data[1], torch.tensor(kmer_data[2])
        ###########Graph Data############
        self.idSeq_virus = []
        for virus in self.k_virus:
            temp_seq = []
            for vir in virus:
                temp_seq.append(self.kmers_dict[vir])
            self.idSeq_virus.append(temp_seq)
        self.idSeq_virus = np.array(self.idSeq_virus, dtype=object)
        # print(f'self.idSeq_virus:{self.idSeq_virus}')
        
        self.idSeq_host = []
        for host in self.k_host:
            temp_seq = []
            for hos in host:
                temp_seq.append(self.kmers_dict[hos])
            self.idSeq_host.append(temp_seq)
        self.idSeq_host = np.array(self.idSeq_host, dtype=object)
        # print(f'self.idSeq_host:{self.idSeq_host}')
        assert len(self.idSeq_host) == len(self.idSeq_virus), 'In ProteinDataset, the num. of host != virus'
        self.vectorize()

        # Construct and save the graph
        self.graph1s = []
        self.graph2s = []
        for i in range(len(self.idSeq_virus)):  
            newidSeq_vir = []  
            newidSeq_hos = []
            old2new_vir = {}  
            old2new_hos = {}
            count_vir = 0
            count_hos = 0
            for eachid in self.idSeq_virus[i]:
                if eachid not in old2new_vir:
                    old2new_vir[eachid] = count_vir
                    count_vir += 1
                newidSeq_vir.append(old2new_vir[eachid])
            counter_uv = Counter(list(zip(newidSeq_vir[:-1], newidSeq_vir[1:]))) 
            # print(f'counter_uv: {counter_uv}')
            graph1 = dgl.graph(list(counter_uv.keys()))
            weight = torch.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph1, weight)
            graph1.edata['weight'] = norm_weight
            node_features = self.vector[list(old2new_vir.keys())]
            graph1.ndata['attr'] = torch.tensor(node_features)
            self.graph1s.append(graph1)
            for eachid in self.idSeq_host[i]:
                if eachid not in old2new_hos:
                    old2new_hos[eachid] = count_hos
                    count_hos += 1
                newidSeq_hos.append(old2new_hos[eachid])
            counter_uv = Counter(list(zip(newidSeq_hos[:-1], newidSeq_hos[1:])))  
            graph2 = dgl.graph(list(counter_uv.keys()))
            weight = torch.FloatTensor(list(counter_uv.values()))
            norm = EdgeWeightNorm(norm='both')
            norm_weight = norm(graph2, weight)
            graph2.edata['weight'] = norm_weight
            node_features = self.vector[list(old2new_hos.keys())]
            graph2.ndata['attr'] = torch.tensor(node_features)
            self.graph2s.append(graph2)

        pad_vir_seqs, len_vir_seqs = self.seq_padding(self.idSeq_virus, self.max_seq_len)
        pad_hos_seqs, len_hos_seqs = self.seq_padding(self.idSeq_host, self.max_seq_len)
        self.virus_seqs, self.len_vir_seqs = torch.from_numpy(pad_vir_seqs), torch.from_numpy(len_vir_seqs)
        self.host_seqs, self.len_hos_seqs = torch.from_numpy(pad_hos_seqs), torch.from_numpy(len_hos_seqs)
        
    def seq_padding(self, seqs, max_len):
        padded_seqs = []
        len_seqs = []
        for seq in seqs:
            new_seq = np.pad(seq, (0, max_len-len(seq)), 'constant', constant_values=(0))
            padded_seqs.append(new_seq)
            len_seqs.append(len(seq))
        return np.array(padded_seqs, dtype=np.int32), np.array(len_seqs)
        
    def __getitem__(self, idx):
        # Get a sample corresponding to it by idx
        return self.graph1s[idx], self.graph2s[idx], self.virus_seqs[idx], self.len_vir_seqs[idx], self.host_seqs[idx], self.len_hos_seqs[idx], self.labels[idx]

    def __len__(self):
        # Number of data samples
        return len(self.graph1s)

    def save(self):
        # Save the processed data to `self.save_path`
        print('***Executing save function***')
        save_graphs(self.save_dir + '_' + params.data_name + '_' + params.neg_type+ '_cf'+ str(params.fold) + "_virus.bin", self.graph1s, {'virus_seqs':self.virus_seqs, 'len_virus_seqs':self.len_vir_seqs})
        save_graphs(self.save_dir + '_' + params.data_name + '_' + params.neg_type+ '_cf'+ str(params.fold) + "_host.bin", self.graph2s, {'labels': self.labels, 'host_seqs':self.host_seqs, 'len_host_seqs':self.len_hos_seqs})
    
    def load(self):
        # Import processed data from `self.save_path`
        print('***Executing load function***')
        try:
            self.graph1s, info_dict_vir = load_graphs(self.save_dir + '_' + params.data_name + '_' + params.neg_type + '_cf'+ str(params.fold) + "_virus.bin")
            self.graph2s, info_dict_hos = load_graphs(self.save_dir + '_' + params.data_name + '_' + params.neg_type + '_cf'+ str(params.fold) + "_host.bin")
            self.labels, self.virus_seqs, self.len_vir_seqs, self.host_seqs, self.len_hos_seqs = info_dict_hos['labels'], info_dict_vir['virus_seqs'], info_dict_vir['len_virus_seqs'], info_dict_hos['host_seqs'], info_dict_hos['len_host_seqs']
        except Exception as e:
            print(f"Failed to load from cache: {e}")
            return False
        return True
    
    def has_cache(self):
        # Check if there is processed data in `self.save_path`
        graph_path_virus = self.save_dir + '_' + params.data_name + '_' + params.neg_type+ '_cf'+ str(params.fold)+"_virus.bin"
        graph_path_host = self.save_dir + '_' + params.data_name + '_' + params.neg_type+ '_cf'+ str(params.fold)+"_host.bin"
        print('fold', str(params.fold), 'gpv', graph_path_virus, 'gph', graph_path_host)
        data_exist = os.path.exists(graph_path_virus) and os.path.exists(graph_path_host)
        if data_exist:
            print(f'Data exist')
        else:
            print(f'No chache data')
        return data_exist
        
    def vectorize(self, method="bert", feaSize=params.d, loadCache=True):
        self.vector = None
        # print('\n***Executing vectorize function***')
        if os.path.exists(
                f'checkpoints/Pair_feature/{params.data_name}_{method}_k{self.kmers}_d{feaSize}_embs.pkl') and loadCache:  # 这一步是看是否存在pkl文件，有的话就直接读取
            with open(f'checkpoints/Pair_feature/{params.data_name}_{method}_k{self.kmers}_d{feaSize}_embs.pkl', 'rb') as f:
                if method == 'word2vec':
                    # print('Loading word2vec....')
                    self.vector = pickle.load(f)
                elif method == 'bert':
                    # print('Loading bert.....')
                    self.vector = pickle.load(f)
                else:
                    print('Kmer embeddings not exist!')

            return