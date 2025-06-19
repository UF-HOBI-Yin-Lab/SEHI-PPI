import torch
import os
from utils.data_single_trans import *
from models.model_single_full_trans import *
from utils.config_single_trans import *
from utils.metrics import *

import datetime
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, model, loaders, optimizer, scheduler, lossfn, params):
        self.gpu_id = params.device
        self.model = model.to(self.gpu_id)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfn = lossfn
        self.params = params
        self.epochs_run = 0
        if not os.path.exists(params.savePath):  # 如果savePath不存在就生成一个
            os.makedirs(params.savePath, exist_ok=True)
        self.snapshot_path = f"%s{self.params.data_name}_{self.params.neg_type}_{self.params.g_model}_{self.params.seq_type}_cf{self.params.fold}_model.pt" % params.savePath
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        print('Load model now')
        # loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["model"])
        self.epochs_run = snapshot["epochs"]

    def _save_snapshot(self, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': self.model.state_dict()}
        torch.save(stateDict, self.snapshot_path)
        print(f"Epoch {epochs} | Training snapshot saved at {self.snapshot_path}")


    def train(self):
        print('Data name:', self.params.data_name, ', Neg Type:', self.params.neg_type, ', gnn:', self.params.g_model, ', seq_type:', self.params.seq_type, ', gpu:', self.params.gpu_mode, ', batch:', self.params.batchSize, ', epoch:', self.params.num_epochs, \
            ', lr:', self.params.lr, ', hid dim:', self.params.hidden_dim)
        train_dataloader, test_dataloader, val_dataloader = self.loaders
                        
        best_record = {'train_loss': 0, 'train_acc': 0, 'train_f': 0,'train_pre': 0,'train_rec': 0, 'train_roc': 0, 'train_sen': 0,'train_spe': 0,'train_mcc': 0, 'train_aps': 0, 'test_loss': 0,  'test_acc': 0, 'test_f': 0,  'test_pre': 0, 'test_rec': 0,'test_roc': 0, 'test_sen': 0,  'test_spe': 0, 'test_mcc': 0,'test_aps': 0}
        nobetter, best_f = 0, 0.0
        for epoch in range(self.epochs_run, self.params.num_epochs):
            train_loss, train_acc, train_f, train_pre, train_rec, train_roc, train_sen, train_spe, train_mcc, train_aps = self.train_epoch(train_dataloader, self.model, self.lossfn, self.optimizer, self.gpu_id)
            val_loss, val_acc, val_f, val_pre, val_rec, val_roc, val_sen, val_spe, val_mcc, val_aps = self.valid_epoch(val_dataloader, self.model, self.lossfn, self.gpu_id)
            self.scheduler.step(val_loss)
            # test_acc, test_f, test_pre, test_rec, test_roc, test_sen, test_spe, test_mcc, test_aps
            print(
                ">>>Epoch:{} of Train Loss:{:.3f}, Valid Loss:{:.3f}\n"
                "Train Acc:{:.3f}, Train F1-score:{:.3f}, Train Precision:{:.3f}, Train Recall:{:.3f}, Train ROC:{:.3f}, Train Sensitivity:{:.3f}, Train Specificity:{:.3f}, Train Matthews Coef:{:.3f}, Train Average Precision:{:.3f};\n"
                "Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid Precision:{:.3f}, Valid Recall:{:.3f}, Valid ROC:{:.3f}, Valid Sensitivity:{:.3f}, Valid Specificity:{:.3f}, Valid Matthews Coef:{:.3f}, Valid Average Precision:{:.3f}!!!\n".format(
                    epoch, train_loss, val_loss,
                    train_acc, train_f, train_pre, train_rec, train_roc, train_sen, train_spe, train_mcc, train_aps,
                    val_acc, val_f, val_pre, val_rec, val_roc, val_sen, val_spe, val_mcc, val_aps))
            if best_f < val_f:
                nobetter = 0
                best_f = val_f
                best_record['train_loss'] = train_loss
                best_record['valid_loss'] = val_loss
                best_record['train_acc'] = train_acc
                best_record['valid_acc'] = val_acc
                best_record['train_f'] = train_f
                best_record['valid_f'] = val_f
                best_record['train_pre'] = train_pre
                best_record['valid_pre'] = val_pre
                best_record['train_rec'] = train_rec
                best_record['valid_rec'] = val_rec
                best_record['train_roc'] = train_roc
                best_record['valid_roc'] = val_roc
                best_record['train_sen'] = train_sen
                best_record['valid_sen'] = val_sen
                best_record['train_spe'] = train_spe
                best_record['valid_spe'] = val_spe
                best_record['train_mcc'] = train_mcc
                best_record['valid_mcc'] = val_mcc
                best_record['train_aps'] = train_aps
                best_record['valid_aps'] = val_aps
                print(f'>Bingo!!! Get a better Model with Valid F1-score: {best_f:.3f}!!!')
                self._save_snapshot(epoch, best_f)
            else:
                nobetter += 1
                if nobetter >= params.earlyStop:
                    print(f'Valid F1-score has not improved for more than {params.earlyStop} steps in epoch {epoch}, stop training.')
                    break
        print("Finally,the model's Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid Precision:{:.3f}, Valid Recall:{:.3f}, Valid ROC:{:.3f}, Valid Sensitivity:{:.3f}, Valid Specificity:{:.3f}, Valid Matthews Coef:{:.3f}, Valid Average Precision:{:.3f}!!!\n\n\n".format(
                best_record['valid_acc'], best_record['valid_f'], best_record['valid_pre'], best_record['valid_rec'], 
                best_record['valid_roc'], best_record['valid_sen'], best_record['valid_spe'], best_record['valid_mcc'], best_record['valid_aps']))
        self.test(test_dataloader, self.model, self.gpu_id)

    def train_epoch(self, train_dataloader, model, loss_fn, optimizer, gpu_id):
        train_loss, train_acc, train_f, train_pre, train_rec, train_roc, train_sen, train_spe, train_mcc, train_aps = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.train()
        pred_list, label_list = [], []
        epoch_loss = 0.0
        num_batches = len(train_dataloader)
        for _, (batched_graph_virus, batched_graph_host, seq_virus, len_virus, seq_host, len_host, labels) in enumerate(train_dataloader):
            # batched_graph_virus, batched_graph_host, seq_virus, seq_host, labels = batched_graph_virus.cuda(gpu_id), batched_graph_host.cuda(gpu_id), seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
            seq_virus, seq_host, labels = seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
            feats1 = batched_graph_virus.ndata['attr'].cuda(gpu_id)
            feats2 = batched_graph_host.ndata['attr'].cuda(gpu_id)
            optimizer.zero_grad()
            # print('model before')
            output = model(batched_graph_virus, feats1, seq_virus, len_virus, batched_graph_host, feats2, seq_host, len_host)
            # print('model after')
            output, labels = output.to(torch.float32), labels.to(torch.int64)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred_list.extend(output.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())

        train_loss = epoch_loss/num_batches
        pred_array = F.softmax(torch.tensor(np.array(pred_list)), dim=1).cpu().numpy()
        labels_list = np.array(label_list)
        preds_list = np.argmax(pred_array, axis=1)
        train_acc, train_f, train_pre, train_rec, train_roc, train_sen, train_spe, train_mcc, train_aps = accuracy(labels_list, preds_list), f1(labels_list, preds_list), precision(labels_list, preds_list), recall(labels_list, preds_list), auc(labels_list, preds_list), sensitivity(labels_list, preds_list), specificity(labels_list, preds_list), mcc(labels_list, preds_list), AUPRC(labels_list, preds_list)
        return train_loss, train_acc, train_f, train_pre, train_rec, train_roc, train_sen, train_spe, train_mcc, train_aps
    
    def valid_epoch(self, test_dataloader, model, loss_fn, gpu_id):
        valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.eval()
        pred_list, label_list = [], []
        epoch_loss = 0.0
        num_batches = len(test_dataloader)
        with torch.no_grad():
            for _, (batched_graph_virus, batched_graph_host, seq_virus, len_virus, seq_host, len_host, labels) in enumerate(test_dataloader):
                # batched_graph_virus, batched_graph_host, seq_virus, seq_host, labels = batched_graph_virus.cuda(gpu_id), batched_graph_host.cuda(gpu_id), seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
                seq_virus, seq_host, labels = seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
                feats1 = batched_graph_virus.ndata['attr'].cuda(gpu_id)
                feats2 = batched_graph_host.ndata['attr'].cuda(gpu_id)
                output = model(batched_graph_virus, feats1, seq_virus, len_virus, batched_graph_host, feats2, seq_host, len_host)
                output, labels = output.to(torch.float32), labels.to(torch.int64)
                loss = loss_fn(output, labels)
                # pred_list loads the predicted values of the training set samples
                pred_list.extend(output.detach().cpu().numpy())
                # label_list loads the true values of the training set samples (one-hot form)
                label_list.extend(labels.cpu().numpy())
                epoch_loss += loss.item()
                    
        valid_loss = epoch_loss/num_batches
        pred_array = F.softmax(torch.tensor(np.array(pred_list)), dim=1).cpu().numpy()
        labels_list = np.array(label_list)
        preds_list = np.argmax(pred_array, axis=1)
        valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps = accuracy(labels_list, preds_list), f1(labels_list, preds_list), precision(labels_list, preds_list), recall(labels_list, preds_list), auc(labels_list, preds_list), sensitivity(labels_list, preds_list), specificity(labels_list, preds_list), mcc(labels_list, preds_list), AUPRC(labels_list, preds_list)
        return valid_loss, valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps
    
    def test(self, eval_dataloader, model, gpu_id):
        model.eval()
        pred_list, label_list = [], []
        with torch.no_grad():
            for _, (batched_graph_virus, batched_graph_host, seq_virus, len_virus, seq_host, len_host, labels) in enumerate(eval_dataloader):
                # batched_graph_virus, batched_graph_host, seq_virus, seq_host, labels = batched_graph_virus.cuda(gpu_id), batched_graph_host.cuda(gpu_id), seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
                seq_virus, seq_host, labels = seq_virus.cuda(gpu_id), seq_host.cuda(gpu_id), labels.cuda(gpu_id)
                feats1 = batched_graph_virus.ndata['attr'].cuda(gpu_id)
                feats2 = batched_graph_host.ndata['attr'].cuda(gpu_id)
                output = model(batched_graph_virus, feats1, seq_virus, len_virus, batched_graph_host, feats2, seq_host, len_host)
                output, labels = output.to(torch.float32), labels.to(torch.int64)
                pred_list.extend(output.detach().cpu().numpy())
                label_list.extend(labels.cpu().numpy())

        pred_array = F.softmax(torch.tensor(np.array(pred_list)), dim=1).cpu().numpy()
        labels_list = np.array(label_list)
        preds_list = np.argmax(pred_array, axis=1)
        valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps = accuracy(labels_list, preds_list), f1(labels_list, preds_list), precision(labels_list, preds_list), recall(labels_list, preds_list), auc(labels_list, preds_list), sensitivity(labels_list, preds_list), specificity(labels_list, preds_list), mcc(labels_list, preds_list), AUPRC(labels_list, preds_list)
        # return valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps
        print("The overall performance on the test data are Acc:{:.3f}, F1-score:{:.3f}, Precision:{:.3f}, Recall:{:.3f}, ROC:{:.3f}, Sensitivity:{:.3f}, Specificity:{:.3f}, Matthews Coef:{:.3f}, Average Precision:{:.3f}!!!\n\n\n".format(
                    valid_acc, valid_f, valid_pre, valid_rec, valid_roc, valid_sen, valid_spe, valid_mcc, valid_aps))

def get_dataloader(dataset, params, train_ratio=0.8, test_ratio=0.1): 
    # 计算每个数据集的大小
    train_size = int(train_ratio * len(dataset))
    val_size = int(test_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # 按照比例划分数据集
    ###step5: 对dataset进行分割,保证每个GPU运行对应的数据
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = GraphDataLoader(train_dataset, batch_size=params.batchSize, shuffle=True,  drop_last=True, num_workers=2)
    val_loader = GraphDataLoader(val_dataset, batch_size=params.batchSize, shuffle=False,  drop_last=False, num_workers=2) 
    test_loader = GraphDataLoader(test_dataset, batch_size=params.batchSize, shuffle=False, drop_last=False, num_workers=2)
    print(f'Training data: {len(train_dataset)}, testing data: {len(test_dataset)}, valid data: {len(val_dataset)}')
    check_lbl('train', train_loader)
    check_lbl('valid', val_loader)
    check_lbl('test', test_loader)
    return [train_loader, test_loader, val_loader]
 
def load_model_objs(params):
    data_name, source_data_name, batch_size, neg_type, seq_type, in_dim, hidden_dim, output_dim, n_classes, g_model, kmer, device  = params.data_name, params.source_data_name, params.batchSize, params.neg_type, params.seq_type, params.d, \
    params.hidden_dim, params.output_dim, params.n_classes, params.g_model, params.k, params.device

    model = PPINet(source_data_name, kmer, seq_type, batch_size, in_dim, hidden_dim, output_dim, n_classes, g_model, device)
    
    # Load the pre-trained model if transfer learning is enabled
    if params.transfer_learning:
        print("Loading pre-trained model for transfer learning...")
        # Load the pre-trained model snapshot
        snapshot_path = f"%s{source_data_name}_{neg_type}_{g_model}_{seq_type}_cf{params.fold}_model.pt" % params.savePath
        snapshot = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(snapshot['model'])

        # Freeze the parameters of seq and gnn modules
        for param in model.seq.parameters():
            param.requires_grad = False

        for param in model.gnn.parameters():
            param.requires_grad = False

        # Modify the classifier head to match the number of classes in dataset B
        model.classify = MLP_layer(
            output_dim * 2, output_dim, n_classes
        ).to(device)

        # Create optimizer with only parameters that require gradients
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params.lr, weight_decay=params.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    lossfn = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, lossfn

def check_lbl(name, dataloader):
    zero, one = 0, 0
    for _, (_, _, _, _, _, _, labels) in enumerate(dataloader):
        one += torch.sum(labels == 1).item()
        zero += torch.sum(labels == 0).item()
    print(f'In {name}, there are {one} 1s, and {zero} 0s.')# Seq for virus are {seq_virs}, and Seq for host are {seq_hoss}')

def main():
    print("Running Python File:", os.path.basename(__file__))
    params = config()
    setup_seed(params.seed)    
    starttime = datetime.datetime.now()
    data_time1 = datetime.datetime.now()
    dataset = ProteinDataset(raw_dir=f'./Dataset/VirusData/{params.data_name}/', save_dir=f'checkpoints/vhgraph/k{params.k}_d{params.d}') #
    loaders = get_dataloader(dataset, params)
    data_time2 = datetime.datetime.now()
    print(f'Data Loading Time is {(data_time2 - data_time1).seconds}s. ')
    train_time1 = datetime.datetime.now()
    model, optimizer, scheduler, lossfn = load_model_objs(params)
    trainer = Trainer(model, loaders, optimizer, scheduler, lossfn, params)
    trainer.train()
    train_time2= datetime.datetime.now()
    print(f'Train time is {(train_time2 - train_time1).seconds}s. ')
    endtime = datetime.datetime.now()
    print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')

if __name__ == '__main__':
    main()