import torch
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl import LaplacianPE

from models.MLP import *
from utils.metrics import *

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

def load_kmers_embs(file):
    with open(file, 'rb') as f:
        embed_matrix = pickle.load(f)
    # print('Kmers embeddings load successfully with file:', file)
    return embed_matrix

def create_emb_layer(kmer, non_trainable=False):
    embs_matrix = torch.from_numpy(load_kmers_embs(f'./checkpoints/Pair_feature/bert_k{kmer}_d1024_embs.pkl'))
    num_embeddings, embedding_dim = embs_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embs_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, embedding_dim
    
    
class PPINet(nn.Module):
    def __init__(self, in_dim, n_classes, device):
        super(PPINet, self).__init__()
        self.in_dim = in_dim
        self.device = device
        self.gnn = GNNModule(in_dim, hidden_dim, g_model, device).to(device)
        self.classify = MLP(inSize=output_dim, outSize=n_classes).to(device)


    def forward(self, g1, h1, g2, h2):
        ####global####
        # sg = self.lstm(x1, len1, x2, len2)
        ####local####
        out = self.gnn(g1, h1, g2, h2)
        # print('out:', out.shape)
        # out = torch.cat((hg, sg), -1)
        return self.classify(out)    
    
    
# 假设我们有一个很大的数据集
X = np.random.randn(1000000, 10)  # 100万个样本，每个样本10个特征
y = np.random.randint(0, 2, 1000000)  # 100万个样本的二分类标签

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义一个函数来训练模型
def train_batch(X_batch, y_batch):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_batch, y_batch)
    return model

# 将数据分块
batch_size = 100000
num_batches = X_train.shape[0] // batch_size

# 使用并行计算训练每个数据块
models = Parallel(n_jobs=-1)(delayed(train_batch)(
    X_train[i * batch_size: (i + 1) * batch_size],
    y_train[i * batch_size: (i + 1) * batch_size]
) for i in range(num_batches))

# 聚合预测结果
def predict_ensemble(models, X):
    # 计算每个模型的预测结果
    predictions = np.array([model.predict(X) for model in models])
    # 聚合结果并返回
    return np.round(predictions.mean(axis=0))

# 验证模型
y_val_pred = predict_ensemble(models, X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Ensemble Random Forest Validation Accuracy: {val_accuracy:.4f}')

# 测试模型
y_test_pred = predict_ensemble(models, X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Ensemble Random Forest Test Accuracy: {test_accuracy:.4f}')


    
