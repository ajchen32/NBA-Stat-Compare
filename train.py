import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Example dataset: Player stats over several games
data = {
    'game': [1, 2, 3, 4, 5],
    'points': [12, 15, 20, 25, 18],
    'assists': [5, 4, 6, 7, 5],
    'rebounds': [10, 8, 7, 9, 6],
    'steals': [2, 3, 1, 2, 3],
}

df = pd.DataFrame(data)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['points', 'assists', 'rebounds', 'steals']])

# Prepare data for LSTM (sequence format)
sequence_length = 3
X = []
y = []

for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length, 0])  # Target: points of the next game

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the LSTM Model
class BasketballLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=50, output_size=1):
        super(BasketballLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Instantiate the model, loss function, and optimizer
model = BasketballLSTM(input_size=4, hidden_layer_size=50, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 200
for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Making predictions
model.eval()
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.detach().numpy()
    print(f"Predicted points for the next game: {predictions}")
