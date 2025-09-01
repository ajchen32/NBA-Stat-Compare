import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

data = np.load('NBA-Stat-Compare/trainandtest_nparraysMOD.npz')

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

print(np.isnan(train_x).any(), np.isnan(train_y).any())
print(np.isnan(test_x).any(), np.isnan(test_y).any())

trainx_tensor = torch.tensor(train_x, dtype=torch.float32)
trainy_tensor = torch.tensor(train_y, dtype=torch.float32)

testx_tensor = torch.tensor(test_x, dtype=torch.float32)
testy_tensor = torch.tensor(test_y, dtype=torch.float32)

print(trainx_tensor.shape)
print(trainy_tensor.shape)

train_dataset = TensorDataset(trainx_tensor, trainy_tensor)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

test_dataset = TensorDataset(testx_tensor, testy_tensor)
test_loader = DataLoader(test_dataset, batch_size = 16)



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
    
input_size = trainx_tensor.shape[2]
hidden_layer = 64
output_size = trainy_tensor.shape[1]
num_layers = 1

print(input_size, hidden_layer, output_size, num_layers)

my_model = LSTMModel(input_size, hidden_layer, output_size, num_layers)
loss_model = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr = .001)

scaler_y = joblib.load("NBA-Stat-Compare/scalerweights.pkl")

epochs = 50
for e in range(epochs):
    for ep_x, ep_y in train_loader:
        optimizer.zero_grad()
        y_pred = my_model(ep_x)
        loss = loss_model(y_pred, ep_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1.0)
        optimizer.step()


    my_model.eval()
    with torch.no_grad():
        test_loss = sum(loss_model(my_model(batch_x), batch_y) for batch_x, batch_y in test_loader) / len(test_loader)
    
    if (e+1) % 10 == 0:
        print(f"Epoch {e+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

        # Prints out results back

        batch_x, batch_y = next(iter(test_loader))  # grab one test batch
        y_pred_scaled = my_model(batch_x).detach().cpu().numpy()
        y_true_scaled = batch_y.detach().cpu().numpy()

        # inverse transform (reshape to 2D for scaler)
        y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
        y_true_orig = scaler_y.inverse_transform(y_true_scaled)

        print("Sample predictions (original scale):")
        for i in range(min(5, len(y_pred_orig))):
            print(f"  Pred: {y_pred_orig[i][0]:.2f}, Actual: {y_true_orig[i][0]:.2f}")


# It works but not optimized yet 

torch.save(my_model.state_dict(), "NBA-Stat-Compare/model.pth")



