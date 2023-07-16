import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Define an autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        
        first_hidden_layer = 64
        second_hidden_layer = 32

        self.encoder = nn.Sequential(
            nn.Linear(input_size, first_hidden_layer),
            nn.ReLU(True),
            nn.Linear(first_hidden_layer, second_hidden_layer),
            nn.ReLU(True),
            nn.Linear(second_hidden_layer, encoding_size))
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, second_hidden_layer),
            nn.ReLU(True),
            nn.Linear(second_hidden_layer, first_hidden_layer),
            nn.ReLU(True),
            nn.Linear(first_hidden_layer, input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
      
      
def create_autoencoder(input_size, encoding_size):
  autoencoder = Autoencoder(input_size, encoding_size)
  return(autoencoder)      

      
def train_autoencoder(autoencoder, df_data, num_epochs = 1000, learning_rate = 0.001, batch_size = 32):
  autoencoder = autoencoder.float()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
  
  epoch_losses = []
  
  train_loader = DataLoader(df_data, batch_size=batch_size)

  for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.float()
        inputs = inputs.view(inputs.size(0), -1)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss/len(train_loader)
    epoch_losses += [epoch_loss]

  return(autoencoder)  



def encode_autoencoder(autoencoder, df_data, batch_size = 32):
  # Encode the synthetic time series data using the trained autoencoder
  train_loader = DataLoader(df_data, batch_size=batch_size)

  encoded_data = []
  for data in train_loader:
      inputs, _ = data
      inputs = inputs.float()
      inputs = inputs.view(inputs.size(0), -1)
      encoded = autoencoder.encoder(inputs)
      encoded_data.append(encoded.detach().numpy())
  
  encoded_data = np.concatenate(encoded_data, axis=0)
  return(encoded_data)


class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_features):
        self.data = np.random.randn(num_samples, seq_length, num_features)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        print(index)
        return self.data[index], self.data[index]
    
# Create the synthetic time series 
num_samples = 1000
seq_length = 6
num_features = 5
#batch_size = 32
data = TimeSeriesDataset(num_samples, seq_length, num_features)
data[:1]

# Train the autoencoder
input_size = num_features * seq_length

autoencoder = create_autoencoder(input_size, 3)

autoencoder = train_autoencoder(autoencoder, data, 1000, 0.001, 32)

encoded_data = encode_autoencoder(autoencoder, data, 32)

