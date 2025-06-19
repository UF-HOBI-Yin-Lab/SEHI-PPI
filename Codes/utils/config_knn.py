import torch as t

#res_rand_orth_gin_h8_o16_b16_e80, res_rand_orth_gin_h16_o32_b32_e80
# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        # k value of k-mer
        self.k = 3
        self.neg_type = 'random'#'random', 'kmers_stat', 'seq_sim' adap_sam
        self.data_name = 'Orthomy'# VirusMentha Orthomy 'VirusMINT', BioGRID, nonIntAct
        self.ml_model = 'knn'
        # Dimension of word2vec word vector, i.e. node feature dimension
        self.d = 1024
        # Parameters of the hidden layer of the graph convolutional networks
        self.num_epochs = 30
        self.earlyStop = 5
        self.lr = 0.001
        # Number of sample categories
        self.n_classes = 2
        # Set random seeds
        self.seed = 42# 42 6657 2024 123 666
        self.fold = 1# 1 2 3 4 5
        # Training parameters
        self.gpu_mode = 'single'
        self.device = t.device("cuda:0")
        self.savePath = f"checkpoints/base/model_ml/mode_{self.gpu_mode}/k{self.k}_d{self.d}/" #这里修改了

