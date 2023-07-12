import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_features):
        self.data = np.random.randn(num_samples, seq_length, num_features)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.data[index]
    
# Create the synthetic time series 
num_samples = 1000
seq_length = 6
num_features = 5
batch_size = 32
data = TimeSeriesDataset(num_samples, seq_length, num_features)

data[:1]

train_loader = DataLoader(data, batch_size=batch_size)

for data in train_loader:
    inputs, outputs = data
    print(inputs.shape)
    print(outputs.shape)
    break

  
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
      
      

# Train the autoencoder
input_size = num_features * seq_length
encoding_size = 3
learning_rate = 0.001
num_epochs = 1000

autoencoder = Autoencoder(input_size, encoding_size)

#autoencoder.apply(weight_init)

autoencoder = autoencoder.float()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

epoch_losses = []

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
  if (epoch + 1) % 10 == 0:
    print('Epoch {} Loss: {:.4f}'.format(epoch+1, epoch_loss))

print("===Finish training===")


# Encode the synthetic time series data using the trained autoencoder
encoded_data = []
for data in train_loader:
    inputs, _ = data
    inputs = inputs.float()
    inputs = inputs.view(inputs.size(0), -1)
    encoded = autoencoder.encoder(inputs)
    encoded_data.append(encoded.detach().numpy())

encoded_data = np.concatenate(encoded_data, axis=0)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2], s=10)
ax.set_xlabel('Latent Dim 1')
ax.set_ylabel('Latent Dim 2')
ax.set_zlabel('Latent Dim 3')
plt.show()



