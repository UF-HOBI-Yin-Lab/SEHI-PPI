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
        return self.linears[1](h)
    
class GNNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, g_model, device):
        super(GNNModule, self).__init__()
        self.g_model = g_model
        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = 1
        self.drop_rate = 0.25

        if self.g_model == 'gin':
            self.conv1 = dglnn.GINConv(MLP_layer(in_dim, hidden_dim, hidden_dim), 'mean', activation=F.relu)
            self.conv2 = dglnn.GINConv(MLP_layer(hidden_dim, hidden_dim, output_dim), 'mean', activation=F.relu)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'gcn':
            #only gcn
            self.conv1 = dglnn.GraphConv(in_feats=in_dim, out_feats=hidden_dim, norm='both', activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = dglnn.GraphConv(in_feats=hidden_dim, out_feats=output_dim, norm='both', activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'graphsage':
            #only graphsage
            self.conv1 = dglnn.SAGEConv(in_feats=in_dim, out_feats=hidden_dim, aggregator_type='mean', feat_drop=0.25, activation=F.relu)
            self.conv2 = dglnn.SAGEConv(in_feats=hidden_dim, out_feats=output_dim, aggregator_type='mean', feat_drop=0.25, activation=F.relu)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'gat':
            #only GAT
            self.conv1 = dglnn.GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=self.num_heads, feat_drop=self.drop_rate, attn_drop=self.drop_rate, activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim*self.num_heads)
            self.conv2 = dglnn.GATConv(in_feats=hidden_dim*self.num_heads, out_feats=output_dim, num_heads=self.num_heads, feat_drop=self.drop_rate, attn_drop=self.drop_rate, activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm2 = nn.BatchNorm1d(output_dim*self.num_heads)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'gcn_gat':
            #GCN + GAT
            self.conv1 = dglnn.GraphConv(in_feats=in_dim, out_feats=hidden_dim, norm='both', activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = dglnn.GATConv(in_feats=hidden_dim, out_feats=output_dim, num_heads=self.num_heads, feat_drop=self.drop_rate, attn_drop=self.drop_rate, activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm2 = nn.BatchNorm1d(output_dim*self.num_heads)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'sgconv':
            #SGConv
            self.conv1 = dglnn.SGConv(in_feats=in_dim, out_feats=hidden_dim, k=2, bias=True, allow_zero_in_degree=True)
            self.conv2 = dglnn.SGConv(in_feats=hidden_dim, out_feats=output_dim, k=2, bias=True, allow_zero_in_degree=True)
            # Dropout and BatchNorm
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'pna':
            #PNA
            self.conv1 = dglnn.PNAConv(in_size=in_dim, out_size=hidden_dim, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification', 'attenuation'], delta=2.5, dropout=0.25)
            self.conv2 = dglnn.PNAConv(in_size=hidden_dim, out_size=output_dim, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification', 'attenuation'], delta=2.5, dropout=0.25)
            # Dropout and BatchNorm
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)

        else:
            print('No Graph Model is assigned')
            import sys
            sys.exit(-1)

    def forward(self, g, h):
        g = g.to(self.device)
        edge_weight = g.edata['weight'].to(self.device)
        if self.g_model in ['gin', 'sgconv']:
            h = F.relu(self.batch_norm1(self.conv1(g, h)))
            h = self.dropout(h)
            h = F.relu(self.batch_norm2(self.conv2(g, h)))
            h = self.dropout(h)
        elif self.g_model in ['gcn', 'graphsage']:
            h = F.relu(self.batch_norm1(self.conv1(g, h, edge_weight=edge_weight)))
            h = self.dropout(h)
            h = F.relu(self.batch_norm2(self.conv2(g, h, edge_weight=edge_weight)))
            h = self.dropout(h)
        elif self.g_model in ['pna']:
            h = F.relu(self.batch_norm1(self.conv1(g, h, edge_feat=edge_weight)))
            h = self.dropout(h)
            h = F.relu(self.batch_norm2(self.conv2(g, h, edge_feat=edge_weight)))
            h = self.dropout(h)
        elif self.g_model in ['gat']:
            #GAT layer 1
            h = self.conv1(g, h)
            h = h.view(-1, self.hidden_dim * self.num_heads)
            h = self.batch_norm1(h)
            h = F.relu(h)  
            h = self.dropout(h)  # Dropout
            #GAT layer 2
            h = self.conv2(g, h)  
            h = h.view(-1, self.output_dim * self.num_heads)  # (batch_size, num_nodes * output_dim * num_heads)
            h = self.batch_norm2(h)  
            h = F.relu(h)  
            h = self.dropout(h)  # Dropout
        elif self.g_model in ['gcn_gat']:
            #GCN layer 1
            h = F.relu(self.batch_norm1(self.conv1(g, h, edge_weight=edge_weight)))
            h = self.dropout(h)
            #GAT layer 2
            h = self.conv2(g, h)  
            h = h.view(-1, self.output_dim * self.num_heads)  # (batch_size, num_nodes * output_dim * num_heads)
            h = self.batch_norm2(h)  
            h = F.relu(h) 
            h = self.dropout(h)  # Dropout

        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        return hg


class LSTMModule(nn.Module):
    """constrcut a LSTM layer to capture the global info""" 
    def __init__(self, data_name, kmer, batch_size, hidden_dim, output_dim, device, num_layers=2, dropout=0.25):
        super(LSTMModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device
        
        self.embedding, self.embedding_dim = create_emb_layer(data_name, kmer, True)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim * 2)
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
        hid_x = torch.mean(hid_x, dim=1)
        output = self.dropout(hid_x)
        return output
    
class GRUModule(nn.Module):
    """constrcut a GRU layer to capture the global info"""
    def __init__(self, data_name, kmer, batch_size, hidden_dim, output_dim, device, num_layers=2, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.embedding, self.embedding_dim = create_emb_layer(data_name, kmer, True)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim * 2)
        self.dropout = nn.Dropout(p=dropout)
        
        
    def init_hidden(self):
        return Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x, len_x):
        self.hidden = self.init_hidden()
        embeds_x = self.embedding(x)
        pad_embeds_x = pack_padded_sequence(embeds_x, len_x.cpu(), batch_first=True, enforce_sorted=False)
        output_x, _ = self.gru(pad_embeds_x, self.hidden)
        hid_x, _ = pad_packed_sequence(output_x, batch_first=True)
        hid_x = torch.mean(hid_x, dim=1)
        output = self.dropout(hid_x)
        return output

class PPINet(nn.Module):
    def __init__(self, data_name, kmer, seq_type, batch_size, in_dim, hidden_dim, output_dim, n_classes, g_model, device):
        super(PPINet, self).__init__()
        if seq_type == 'lstm':
            self.seq = LSTMModule(data_name, kmer, batch_size, hidden_dim, output_dim, device).to(device)
        elif seq_type == 'gru':
            self.seq = GRUModule(data_name, kmer, batch_size, hidden_dim, output_dim, device).to(device)
        self.gnn = GNNModule(in_dim, hidden_dim, int(output_dim/2), g_model, device).to(device)
        
        self.classify = MLP_layer(output_dim*2, output_dim, n_classes).to(device)
        
    def get_embeds(self, g1, h1, x1, len1, g2, h2, x2, len2, return_full=False):
        ####global####
        sg1 = self.seq(x1, len1)
        sg2 = self.seq(x2, len2)
        sg = torch.cat((sg1, sg2), 1)
        ####local####
        hg1 = self.gnn(g1, h1)
        hg2 = self.gnn(g2, h2)
        hg = torch.cat((hg1, hg2), -1)
        out = torch.cat((hg, sg), -1)
        if return_full:
            embs_virus = torch.cat((sg1, hg1), dim=1)
            embs_host = torch.cat((sg2, hg2), dim=1)
            return embs_virus, embs_host, out
        else:
            return out
    
    def forward(self, g1, h1, x1, len1, g2, h2, x2, len2):
        ####global####
        sg1 = self.seq(x1, len1)
        sg2 = self.seq(x2, len2)
        sg = torch.cat((sg1, sg2), 1)
        ####local####
        hg1 = self.gnn(g1, h1)
        hg2 = self.gnn(g2, h2)
        hg = torch.cat((hg1, hg2), -1)
        out = torch.cat((hg, sg), -1)
        pred = self.classify(out)
        return pred