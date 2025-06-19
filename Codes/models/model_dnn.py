import torch
import torch.nn as nn
import torch.optim as optim

class DNPPPIModel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=64, num_filters=10):
        super(DNPPPIModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=embedding_dim, padding_idx=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 2 sequences concatenated
    

    def forward(self, seq1, seq2):
        # Embedding
        emb1 = self.embedding(seq1).permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        emb2 = self.embedding(seq2).permute(0, 2, 1)
        # print('emb1', emb1.shape, 'emb2', emb2.shape)

        # CNN
        def apply_cnn(x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            return x.permute(0, 2, 1)  # [batch_size, seq_len, num_filters]
        
        cnn1 = apply_cnn(emb1)
        cnn2 = apply_cnn(emb2)
        # print('cnn1', cnn1.shape, 'cnn2', cnn2.shape)
        
        # LSTM
        lstm1, _ = self.lstm(cnn1)  # [batch_size, seq_len, hidden_dim]
        lstm2, _ = self.lstm(cnn2)  # [batch_size, seq_len, hidden_dim]
        # print('lstm1', lstm1.shape, 'lstm2', lstm2.shape)
        
        # 使用最后一个时间步的隐藏状态
        lstm1_last = lstm1[:, -1, :]  # [batch_size, hidden_dim]
        lstm2_last = lstm2[:, -1, :]  # [batch_size, hidden_dim]
        
        # Fully connected
        combined = torch.cat((lstm1_last, lstm2_last), dim=1)  # [batch_size, hidden_dim * 2]
        # print('combined', combined.shape)
        out = self.fc(combined)  # [batch_size, 1]
        # print('out', out.shape)
        return out