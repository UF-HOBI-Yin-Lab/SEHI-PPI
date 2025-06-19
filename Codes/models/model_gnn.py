import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import LaplacianPE
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable
import pickle

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
            # self.conv = dglnn.GINConv(MLP_layer(in_dim, hidden_dim, output_dim), 'mean', activation=F.relu)
            # self.batch_norm = nn.BatchNorm1d(output_dim)
            # self.dropout = nn.Dropout(p=self.drop_rate)
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
            # 批标准化和Dropout
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
            # Dropout 和 BatchNorm
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'pna':
            #PNA
            self.conv1 = dglnn.PNAConv(in_size=in_dim, out_size=hidden_dim, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification', 'attenuation'], delta=2.5, dropout=0.25)
            self.conv2 = dglnn.PNAConv(in_size=hidden_dim, out_size=output_dim, aggregators=['mean', 'max', 'sum'], scalers=['identity', 'amplification', 'attenuation'], delta=2.5, dropout=0.25)
            # Dropout 和 BatchNorm
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
            # h = F.relu(self.batch_norm(self.conv(g, h)))
            # h = self.dropout(h)
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
            h = self.batch_norm1(h)# 批标准化
            h = F.relu(h)  # 激活函数
            h = self.dropout(h)  # Dropout
            #GAT layer 2
            h = self.conv2(g, h)  
            h = h.view(-1, self.output_dim * self.num_heads)  # 重塑为 (batch_size, num_nodes * output_dim * num_heads)
            h = self.batch_norm2(h)  # 批标准化
            h = F.relu(h)  # 激活函数
            h = self.dropout(h)  # Dropout
        elif self.g_model in ['gcn_gat']:
            #GCN layer 1
            h = F.relu(self.batch_norm1(self.conv1(g, h, edge_weight=edge_weight)))
            h = self.dropout(h)
            #GAT layer 2
            h = self.conv2(g, h)  
            h = h.view(-1, self.output_dim * self.num_heads)  # 重塑为 (batch_size, num_nodes * output_dim * num_heads)
            h = self.batch_norm2(h)  # 批标准化
            h = F.relu(h)  # 激活函数
            h = self.dropout(h)  # Dropout

        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
        return hg
    
class PPINet(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, n_classes, g_model, device):
        super(PPINet, self).__init__()
        self.gnn = GNNModule(in_dim, hidden_dim, output_dim, g_model, device).to(device)
        self.classify = MLP_layer(output_dim, int(output_dim/2), n_classes).to(device)
        # self.classify = MLP_layer(output_dim*2, output_dim, n_classes).to(device)
        # self.classify = nn.Linear(output_dim*2, n_classes).to(device)

    def get_embeds(self, g1, h1, g2, h2):
        ####local####
        hg1 = self.gnn(g1, h1)
        hg2 = self.gnn(g2, h2)
        hg = torch.cat((hg1, hg2), -1)
        # print('hg', hg.shape, hg1.shape, hg2.shape)
        return hg

    def forward(self, g1, h1, g2, h2):
        ####local####
        hg1 = self.gnn(g1, h1)
        hg2 = self.gnn(g2, h2)
        hg  = torch.mean(torch.stack([hg1, hg2]), dim=0)
        # hg = torch.cat((hg1, hg2), -1)
        # print('hg', hg.shape, 'hg1', hg1.shape, 'hg2', hg2.shape)
        pred = self.classify(hg)
        # print('pred', pred.shape)
        # import sys
        # sys.exit(-1)
        return pred

        