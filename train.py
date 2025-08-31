import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = np.load('NBA-Stat-Compare/trainandtest_nparrays.npz')
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']


trainx_tensor = torch.tensor(train_x, dtype=torch.float32)
trainy_tensor = torch.tensor(train_y, dtype=torch.float32).view(-1, 1)


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
    
input_size = 
hidden_layer = 
output_size = 
num_layers = 

my_model = LSTMModel(input_size, hidden_layer, output_size, num_layers)
loss_model = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr = .001)

epochs = 50

# Define the LSTM Model
# class BasketballLSTM(nn.Module):
#     def __init__(self, input_size=4, hidden_layer_size=50, output_size=1):
#         super(BasketballLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Linear(hidden_layer_size, output_size)
    
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         predictions = self.linear(lstm_out[:, -1, :])
#         return predictions

# # Instantiate the model, loss function, and optimizer
# model = BasketballLSTM(input_size=4, hidden_layer_size=50, output_size=1)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training the model
# epochs = 200
# for epoch in range(epochs):
#     model.train()
#     y_pred = model(X)
#     loss = loss_function(y_pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# # Making predictions
# model.eval()
# with torch.no_grad():
#     predictions = model(X)
#     predictions = predictions.detach().numpy()
#     print(f"Predicted points for the next game: {predictions}")
