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
        
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        # self.sag = SAGPooling(hidden_dim, 0.5)
        
        if self.g_model == 'gin':
            self.conv1 = dglnn.GINConv(MLP_layer(in_dim, hidden_dim, hidden_dim), 'mean', activation=F.relu)
            self.conv2 = dglnn.GINConv(MLP_layer(hidden_dim, hidden_dim, output_dim), 'mean', activation=F.relu)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        elif self.g_model == 'gcn':
            #only gcn
            self.conv1 = dglnn.GraphConv(in_feats=in_dim, out_feats=in_dim, norm='both', activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm1 = nn.BatchNorm1d(in_dim)
            self.conv2 = dglnn.GraphConv(in_feats=in_dim, out_feats=in_dim, norm='both', activation=F.relu, allow_zero_in_degree=True)
            self.batch_norm2 = nn.BatchNorm1d(in_dim)
            self.dropout = nn.Dropout(p=self.drop_rate)
        else:
            print('No Graph Model is assigned')
            import sys
            sys.exit(-1)

    def forward(self, g, h, final=True):
        g = g.to(self.device)
        edge_weight = g.edata['weight'].to(self.device)
        if self.g_model in ['gin']:
            h = F.relu(self.batch_norm1(self.conv1(g, h)))
            h = self.dropout(h)
            h = F.relu(self.batch_norm2(self.conv2(g, h)))
            h = self.dropout(h)
        elif self.g_model in ['gcn']:
            h = self.conv1(g, h, edge_weight=edge_weight)
            # print('hh', h.shape)#hh torch.Size([12581, 16])
            h = self.fc1(h)
            # print('hh1', h.shape)
            h = F.relu(h)
            h = self.batch_norm1(h)
            h = self.dropout(h)
            # print('h0', h.shape)
            h = self.conv2(g, h, edge_weight=edge_weight)
            h = self.fc2(h)
            h = F.relu(h)
            h = self.batch_norm2(h)
            h = self.dropout(h)
            # print('h1', h.shape)

        if final:
            with g.local_scope():
                g.ndata['h'] = h
                hg = dgl.mean_nodes(g, 'h')
            return hg
        else:
            return h
        
class PPINet(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, n_classes, device):
        super(PPINet, self).__init__()
        self.gcn = GNNModule(in_dim, hidden_dim, int(output_dim/2), 'gcn', device).to(device)
        self.gin = GNNModule(in_dim, hidden_dim, int(output_dim/2), 'gin', device).to(device)
        self.classify = MLP_layer(int(output_dim/2), int(output_dim/4), n_classes).to(device)
        # self.classify = MLP_layer(output_dim*2, output_dim, n_classes).to(device)
        # self.classify = nn.Linear(output_dim*2, n_classes).to(device)


    def forward(self, g1, h1, g2, h2):
        ####local####
        # print('ss', h1.shape) #torch.Size([12692, 1024])
        hg1 = self.gcn(g1, h1, False)
        # print('hg1-1', hg1.shape) #hg1-1 torch.Size([12805, 1024])
        hg1 = self.gin(g1, hg1) 
        # print('hg1-2', hg1.shape)#hg1-2 torch.Size([32, 16])
        hg2 = self.gcn(g2, h2, False)
        # print('hg2-1', hg2.shape)#hg2-1 torch.Size([10850, 1024])
        hg2 = self.gin(g2, hg2)
        # print('hg2-2', hg2.shape)#hg2-2 torch.Size([32, 16])
        hg  = torch.mean(torch.stack([hg1, hg2]), dim=0)
        # hg = torch.cat((hg1, hg2), -1)
        # print('hg', hg.shape, 'hg1', hg1.shape, 'hg2', hg2.shape)#hg torch.Size([32, 16]) hg1 torch.Size([32, 16]) hg2 torch.Size([32, 16])
        pred = self.classify(hg)
        # print('pred', pred.shape)
        return pred
