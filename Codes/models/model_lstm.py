import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import LaplacianPE
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable
import pickle

def load_kmers_embs(file):
    with open(file, 'rb') as f:
        embed_matrix = pickle.load(f)
    # print('Kmers embeddings load successfully with file:', file)
    return embed_matrix

def create_emb_layer(data_name, kmer, trainable=True):
    embs_matrix = torch.from_numpy(load_kmers_embs(f'./checkpoints/Pair_feature/{data_name}_bert_k{kmer}_d1024_embs.pkl'))
    num_embeddings, embedding_dim = embs_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embs_matrix})
    if trainable:
        emb_layer.weight.requires_grad = True
    return emb_layer, embedding_dim

class MLP_layer(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=True))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=True))
        self.dropout = nn.Dropout(p=0.25)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, h):
        h = F.relu(self.batch_norm(self.linears[0](h)))
        h = self.dropout(h)
        # h = F.relu(self.linears[0](h))
        return self.linears[1](h)

class LSTMModule(nn.Module):
    """constrcut a LSTM layer to capture the global info""" 
    def __init__(self, data_name, kmer, batch_size, hidden_dim, output_dim, device, num_layers=1, dropout=0.25):
        super(LSTMModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device
        
        # 加载K-mer嵌入层
        self.embedding, self.embedding_dim = create_emb_layer(data_name, kmer, True)
        # LSTM层，内部包含Dropout
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        # 输出层
        # self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim * 2)
        self.dropout = nn.Dropout(p=dropout)
        
    def init_hidden(self):
        return (Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_dim).to(self.device)),
                Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_dim).to(self.device)))
        
    def forward(self, x, len_x):
        self.hidden = self.init_hidden()
        embeds_x = self.embedding(x)
        pad_embeds_x = pack_padded_sequence(embeds_x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        output_x, _ = self.lstm(pad_embeds_x, self.hidden)
        hid_x, _ = pad_packed_sequence(output_x, batch_first=True)
        # 序列的平均池化
        # print('hid_x', hid_x.shape)
        # hid_x = torch.mean(hid_x, dim=1)
        # print('hid_x1', hid_x.shape)
        output = self.dropout(hid_x)
        output = torch.mean(hid_x, dim=1)
        # output = hid_x[:, -1, :]
        # print('hid_x', hid_x.shape, 'tt', tt.shape)
        return output
    
class PPINet(nn.Module):
    def __init__(self, data_name, kmer, seq_type, batch_size, hidden_dim, output_dim, n_classes, device):
        super(PPINet, self).__init__()
        if seq_type == 'lstm':
            self.seq = LSTMModule(data_name, kmer, batch_size, hidden_dim, output_dim, device).to(device)
        elif seq_type == 'gru':
            pass
        #     self.seq = GRUModule(data_name, kmer, batch_size, hidden_dim, output_dim, device).to(device)
        # self.classify = MLP_layer(int(output_dim/2), int(output_dim/4), n_classes).to(device)
        self.classify = nn.Linear(int(output_dim/2), n_classes).to(device)

    def get_embeds(self, x1, len1, x2, len2):
        ####global####
        sg1 = self.seq(x1, len1)
        sg2 = self.seq(x2, len2)
        sg = torch.cat((sg1, sg2), 1)
        return sg

    def forward(self, x1, len1, x2, len2):
        ####global####
        sg1 = self.seq(x1, len1)
        sg2 = self.seq(x2, len2)
        # sg = torch.cat((sg1, sg2), 1)
        # print('sg1:',sg1.shape, 'sg2:', sg2.shape, 'sg:', sg.shape)
        sg  = torch.mean(torch.stack([sg1, sg2]), dim=0)
        # print('sg', sg.shape, 'sg1', sg1.shape, 'sg2', sg2.shape)
        pred = self.classify(sg)
        # print('pred', pred.shape)
        # import sys
        # sys.exit(-1)
        return pred
