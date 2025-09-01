import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)

        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

# use the testing batch for more close analysis
data = np.load('NBA-Stat-Compare/trainandtest_nparraysMOD.npz')
test_x = data['test_x']
test_y = data['test_y']

testx_tensor = torch.tensor(test_x, dtype=torch.float32)
testy_tensor = torch.tensor(test_y, dtype=torch.float32)

input_size = testx_tensor.shape[2]
hidden_layer = 64
output_size = testy_tensor.shape[1]
num_layers = 1

loading_model = LSTMModel(input_size, hidden_layer, output_size, num_layers)

# load the model from train.py
loading_model.load_state_dict(torch.load("NBA-Stat-Compare/model.pth"))

