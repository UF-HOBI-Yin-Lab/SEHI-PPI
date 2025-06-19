import torch
import os
from utils.data_single_knn import *
# from utils.config_decisiontree import *
# from utils.config_randomforest import *
# from utils.config_xgboost import *
# from utils.config_sgd import *
# from utils.config_svm import *
from utils.config_knn import *
# from utils.config_gnb import *

import datetime
from utils.metrics import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.data import random_split
# from dgl.dataloading import GraphDataLoader
from torch.utils.data import DataLoader


import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def setup_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, loaders, params):
        self.loaders = loaders
        self.params = params        
        if not os.path.exists(self.params.savePath):  # 如果savePath不存在就生成一个
            os.makedirs(self.params.savePath, exist_ok=True)
        self.suffix = 'model' if self.params.ml_model == 'xgboost' else 'joblib'
        self.snapshot_path = f"%s{self.params.data_name}_{self.params.neg_type}_{self.params.ml_model}_cf{self.params.fold}_model.{self.suffix}" % self.params.savePath
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        print('Load model now')
        if self.params.ml_model == 'xgboost':
            # 加载模型
            model = lgb.Booster(model_file=snapshot_path)
        else:
            model = joblib.load(snapshot_path)
        return model

    def _save_snapshot(self, model, snapshot_path):
        if self.params.ml_model == 'xgboost':
            model.save_model(snapshot_path)
        else:
            joblib.dump(model, snapshot_path)
        print('Save model now')
        
    def _process_dataloader(self, dataloader):
        pair_embs, labels = next(iter(dataloader))
        return pair_embs, labels
    
    def model_train(self, loaders):
        train_dataloader, test_dataloader, valid_dataloader = loaders
        print('Data name:', self.params.data_name, ', Neg Type:', self.params.neg_type, ', model_name:', self.params.ml_model, ', gpu:', self.params.gpu_mode)
        if self.params.ml_model == 'decisiontree':
            model = DecisionTreeClassifier(max_depth=20, min_samples_split=100, random_state=self.params.seed)
        elif self.params.ml_model == 'randforest':
            model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=100, n_jobs=-1, random_state=self.params.seed)
        elif self.params.ml_model == 'sgd':
            model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        elif self.params.ml_model == 'svm':
            model = LinearSVC(max_iter=10000)
        elif self.params.ml_model == 'knn':
            model = KNeighborsClassifier(n_neighbors=3, algorithm='auto', n_jobs=-1)
        elif self.params.ml_model == 'gnb':
            model = GaussianNB()
        elif self.params.ml_model == 'xgboost':
            pass
        else:
            model = None
            assert f'No {self.params.ml_model} model has been trained'        
            
        # 将训练集加载到内存中进行模型训练
        train_embs, train_labels = self._process_dataloader(train_dataloader)
        val_embs, val_labels = self._process_dataloader(valid_dataloader)
        test_embs, test_labels = self._process_dataloader(test_dataloader)
        if self.params.ml_model == 'xgboost':
            xgb_params = {
                'objective': 'binary',  # Binary classification
                'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
                'metric': 'binary_logloss',  # Evaluation metric
                'num_leaves': 15,  # Number of leaves in one tree
                'learning_rate': self.params.lr,  # Learning rate
                'feature_fraction': 0.9,  # Fraction of features to use for training
                'min_data_in_leaf': 20,  # Increase to prevent overfitting
                'lambda_l1': 1.0,  # Add L1 regularization
                'lambda_l2': 1.0   # Add L2 regularization
                }
    
            # Create the LightGBM dataset
            train_data = lgb.Dataset(train_embs, label=train_labels.tolist())
            valid_data = lgb.Dataset(val_embs, label=val_labels.tolist(), reference=train_data)
            # Train the model
            model = lgb.train(
                xgb_params,
                train_data,
                num_boost_round=100,  # Maximum number of boosting rounds
                valid_sets=[train_data, valid_data],  # Validation data for early stopping
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
            )
            # Make predictions on the test set
            test_pred = model.predict(test_embs, num_iteration=model.best_iteration)
            # Convert probabilities to binary predictions
            test_pred = [1 if x > 0.5 else 0 for x in test_pred]
            test_acc, test_f, test_pre, test_rec, test_roc, test_sen, test_spe, test_mcc, test_aps = accuracy(test_labels, test_pred), f1(test_labels, test_pred), precision(test_labels, test_pred), recall(test_labels, test_pred), auc(test_labels, test_pred), sensitivity(test_labels, test_pred), specificity(test_labels, test_pred), mcc(test_labels, test_pred), AUPRC(test_labels, test_pred)
            print("Test Acc:{:.3f}, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}, Test Sensitivity:{:.3f}, Test Specificity:{:.3f}, Test Matthews Coef:{:.3f}, Test Average Precision:{:.3f}!!!\n"
                .format(test_acc, test_f, test_pre, test_rec, test_roc, test_sen, test_spe, test_mcc, test_aps))
        else:
            model.fit(train_embs, train_labels)
            val_pred = model.predict(val_embs)
            valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps = accuracy(val_labels, val_pred), f1(val_labels, val_pred), precision(val_labels, val_pred), recall(val_labels, val_pred), auc(val_labels, val_pred), sensitivity(val_labels, val_pred), specificity(val_labels, val_pred), mcc(val_labels, val_pred), AUPRC(val_labels, val_pred)
            test_pred = model.predict(test_embs)
            test_acc, test_f, test_pre, test_rec, test_roc, test_sen, test_spe, test_mcc, test_aps = accuracy(test_labels, test_pred), f1(test_labels, test_pred), precision(test_labels, test_pred), recall(test_labels, test_pred), auc(test_labels, test_pred), sensitivity(test_labels, test_pred), specificity(test_labels, test_pred), mcc(test_labels, test_pred), AUPRC(test_labels, test_pred)
            print("Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid Precision:{:.3f}, Valid Recall:{:.3f}, Valid ROC:{:.3f}, Valid Sensitivity:{:.3f}, Valid Specificity:{:.3f}, Valid Matthews Coef:{:.3f}, Valid Average Precision:{:.3f};\n"
              "Test Acc:{:.3f}, Test F1-score:{:.3f}, Test Precision:{:.3f}, Test Recall:{:.3f}, Test ROC:{:.3f}, Test Sensitivity:{:.3f}, Test Specificity:{:.3f}, Test Matthews Coef:{:.3f}, Test Average Precision:{:.3f}!!!\n".format(
                  valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps,
                    test_acc, test_f, test_pre, test_rec, test_roc, test_sen, test_spe, test_mcc, test_aps
                    ))
        self._save_snapshot(model, self.snapshot_path)
        
def check_lbl(name, dataloader):
    zero, one = 0, 0
    for _, (_, labels) in enumerate(dataloader):
        one += torch.sum(labels == 1).item()
        zero += torch.sum(labels == 0).item()
    print(f'In {name}, there are {one} 1s, and {zero} 0s.')# Seq for virus are {seq_virs}, and Seq for host are {seq_hoss}')
    
def get_dataloader(dataset, train_ratio=0.8, test_ratio=0.1): 
    # 计算每个数据集的大小
    train_size = int(train_ratio * len(dataset))
    val_size = int(test_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # 按照比例划分数据集
    ###step5: 对dataset进行分割,保证每个GPU运行对应的数据
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True,  drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False,  drop_last=False, num_workers=0) 
    print(f'Training data: {len(train_dataset)}, testing data: {len(test_dataset)}, valid data: {len(val_dataset)}')
    check_lbl('train', train_loader)
    check_lbl('valid', val_loader)
    check_lbl('test', test_loader)
    return [train_loader, test_loader, val_loader]

def main():
    print("Running Python File:", os.path.basename(__file__))
    params = config()
    setup_seed(params.seed)
    starttime = datetime.datetime.now()
    data_time1 = datetime.datetime.now()
    dataset = ProteinDataset(raw_dir=f'./Dataset/{params.data_name}/', save_dir=f'checkpoints/vhgraph/ML/') #
    loaders = get_dataloader(dataset)
    data_time2 = datetime.datetime.now()
    print(f'Data Loading Time is {(data_time2 - data_time1).seconds}s. ')
    train_time1 = datetime.datetime.now()
    trainer = Trainer(loaders, params)
    trainer.model_train(loaders)
    train_time2= datetime.datetime.now()
    print(f'Train time is {(train_time2 - train_time1).seconds}s. ')
    endtime = datetime.datetime.now()
    print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')


if __name__ == '__main__':
    main()