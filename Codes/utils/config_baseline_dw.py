import torch as t

#res_rand_orth_gin_h8_o16_b16_e80, res_rand_orth_gin_h16_o32_b32_e80
# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        # k value of k-mer
        self.k = 3
        self.neg_type = 'adap_sam'#'random', 'kmers_stat', 'seq_sim' adap_sam
        self.seq_type = 'lstm' # lstm gru
        self.data_name = 'Orthomy'# VirusMentha Orthomy VirusMentha Orthomy 'VirusMINT', BioGRID, nonIntAct
        self.embs_type = 'deepwalk'
        # Dimension of word2vec word vector, i.e. node feature dimension
        self.d = 64
        # Parameters of the hidden layer of the graph convolutional networks
        self.hidden_dim = 16#8#128#128
        self.output_dim = 32#16#64
        # Number of sample categories
        self.n_classes = 2
        # Set random seeds
        self.seed = 42# 42 6657 2024 123 666
        self.fold = 1# 1 2 3 4 5
        # Training parameters
        self.batchSize = 32#16#32#64
        self.num_epochs = 80#5#80#150
        self.lr = 0.001#0.001
        self.weight_decay = 0.001
        self.earlyStop = 20
        self.g_model = 'gin' #gin, pna, dgn, gcn2conv
        self.gpu_mode = 'single'
        self.savePath = f"checkpoints/base/model_{self.g_model}/mode_{self.gpu_mode}/k{self.k}_d{self.d}_h{self.hidden_dim}_b{self.batchSize}/" #这里修改了
        # self.run_method = 'nm' #mp: multiprocessing, nm: normal
        self.device = t.device("cuda:0")