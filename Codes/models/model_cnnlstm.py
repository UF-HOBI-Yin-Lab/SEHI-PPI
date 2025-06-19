# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class PPIModel(nn.Module):
#     def __init__(self):
#         super(PPIModel, self).__init__()

#         # 卷积层
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)

#         # 最大池化层
#         self.maxpool = nn.MaxPool1d(kernel_size=2)

#         # 全连接层
#         self.fc1 = nn.Linear(96, 64)  # 需要根据输入长度计算
#         self.fc2 = nn.Linear(64, 1)

#         # Dropout
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
#         x = x.unsqueeze(1)  # (batch_size, 1, seq_len)

#         x = F.relu(self.conv1(x))  # (batch_size, 16, seq_len)
#         x = self.maxpool(x)        # (batch_size, 16, seq_len/2)

#         x = F.relu(self.conv2(x))  # (batch_size, 32, seq_len/2)
#         x = self.maxpool(x)        # (batch_size, 32, seq_len/4)

#         x = x.view(x.size(0), -1)  # 展平特征 (batch_size, 32 * (seq_len/4))

#         x = F.relu(self.fc1(x))    # (batch_size, 64)
#         x = self.dropout(x)
#         x = self.fc2(x)  # (batch_size, 1)

#         return x


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class PPIModel(nn.Module):
    def __init__(self):
        super(PPIModel, self).__init__()

        # 卷积层1: 输入 (16, 1, 472)，输出 (16, 1, 474)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=15, padding=8)

        # 卷积层2: 输出 (16, 1, 244)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, padding=8)

        # 卷积层3: 输出 (16, 1, 131)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8, padding=8)

        # 卷积层4: 输出 (16, 1, 39)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=8)

        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # 双向 LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=80, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc1 = nn.Linear(160, 64)  # LSTM 是双向的，因此输出是 80 * 2 = 160
        self.fc2 = nn.Linear(64, 1)  # 输出为 1 个分类结果

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 输入形状为 (batch_size, 1, 472)
        # print('x:', x.shape)
        x = x.unsqueeze(1)
        # print('x1:', x.shape)
        x = F.relu(self.conv1(x))  # 输出形状 (batch_size, 1, 474)
        # print('1:', x.shape)
        x = self.maxpool(x)        # 池化后的输出 (batch_size, 1, 237)
        # print('2:', x.shape)  
        x = F.relu(self.conv2(x))  # 输出形状 (batch_size, 1, 244)
        # print('3:', x.shape)  
        x = self.maxpool(x)        # 池化后的输出 (batch_size, 1, 122)
        # print('4:', x.shape)
        x = F.relu(self.conv3(x))  # 输出形状 (batch_size, 1, 131)
        # print('5:', x.shape)  
        x = self.maxpool(x)        # 池化后的输出 (batch_size, 1, 66)
        # print('6:', x.shape)
        x = F.relu(self.conv4(x))  # 输出形状 (batch_size, 1, 39)
        # print('7:', x.shape)
        x = self.maxpool(x)        # 池化后的输出 (batch_size, 1, 19)
        # print('8:', x.shape)
        # 调整输入形状以适配 LSTM (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # 将形状从 (batch_size, 1, 19) 转置为 (batch_size, 19, 1)
        # print('9:', x.shape)
        # LSTM 层
        x, _ = self.lstm(x)  # 输出形状 (batch_size, 19, 160)
        # print('10:', x.shape)
        # 取 LSTM 的最后一个时间步的输出
        x = x[:, -1, :]  # 最后一个时间步的 hidden state，形状 (batch_size, 160)
        # print('11:', x.shape)
        # 全连接层
        x = F.relu(self.fc1(x))  # 输出形状 (batch_size, 64)
        # print('12:', x.shape)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 输出形状 (batch_size, 1)
        # print('13:', x.shape)
        return x
