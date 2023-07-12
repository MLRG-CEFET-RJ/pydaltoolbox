import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from train import fit

class LSTMNet(nn.Module):
    def __init__(self, n_neurons, input_shape):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=n_neurons)
        self.fc = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

def savemodel(model, filename):
  torch.save(model, filename)
  

def loadmodel(filename):
  model = torch.load(filename)
  model.eval()
  return(model)


def savedf(data, filename):      
    data.to_csv(filename, index=False)
    
    
def create_model(n_neurons, look_back):
  criterion = nn.MSELoss()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # model = model.float()
  model = LSTMNet(n_neurons, look_back).to(device)
  return model  
    
def train_pytorch(model, df_train, df_val, filename, n_epochs = 3000):
    X_train = df_train.drop('t0', axis=1).to_numpy()
    y_train = df_train.t0.to_numpy()
    X_train = X_train[:, :, np.newaxis]
    y_train = y_train[:, np.newaxis]	
    train_x = torch.from_numpy(X_train)
    train_y = torch.from_numpy(y_train)
    train_x = torch.permute(train_x, (2, 0, 1))
    train_labels = torch.permute(train_y, (1, 0))
    train_labels = train_labels[:, :, None]
    train_ds = TensorDataset(train_x, train_labels)

    X_val = df_val.drop('t0', axis=1).to_numpy()
    y_val = df_val.t0.to_numpy()
    X_val = X_val[:, :, np.newaxis]
    y_val = y_val[:, np.newaxis]
    valid_x = torch.from_numpy(X_val)
    valid_y = torch.from_numpy(y_val)
    valid_x = torch.permute(valid_x, (2, 0, 1))
    val_labels = torch.permute(valid_y, (1, 0))
    val_labels = val_labels[:, :, None]
    valid_ds = TensorDataset(valid_x, val_labels)

    BATCH_SIZE = 8
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle = False)

    PATIENCE = 20
    model = model.float()
    criterion = nn.MSELoss()
    
    train_loss, valid_loss = fit(model, train_loader, valid_loader, n_epochs, lr = 0.001, criterion = criterion, patience = PATIENCE, filename = filename, opt_func=torch.optim.Adam)

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot-conv1d.png', bbox_inches='tight')
    return model


def predict_pytorch(model, df_test):
  X_test = df_test.drop('t0', axis=1).to_numpy()
  y_test = df_test.t0.to_numpy()
  X_test = X_test[:, :, np.newaxis]
  y_test = y_test[:, np.newaxis]
  test_x = torch.from_numpy(X_test)
  test_labels = torch.from_numpy(y_test)
  test_x = torch.permute(test_x, (2, 0, 1))	
  test_labels = torch.permute(test_labels, (1, 0))
  test_labels = test_labels[:, :, None]
  test_ds = TensorDataset(test_x, test_labels)
  
  BATCH_SIZE = 8
  test_loader = torch.utils.data.DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False)
  
  outputs = []
  with torch.no_grad():
    for xb, yb in test_loader:
      output = model(xb.float())
      outputs.append(output)
  
  test_predictions = torch.vstack(outputs).squeeze(1)  
  test_predictions = test_predictions.numpy()
  
  return test_predictions


df_train = pd.read_csv('./data/train_sin.csv')
df_val = pd.read_csv('./data/val_sin.csv')
df_test = pd.read_csv('./data/test_sin.csv')


TRAIN_MODE = True
FILENAME = 'lstm.model'

if TRAIN_MODE:
  print('Creating the model...')
  model = create_model(n_neurons = 4, look_back = 4)
  print(model)
  print('Done!')
  print('Training the model...')
  train_pytorch(model, df_train, df_val, FILENAME, n_epochs = 3000)
  print('Done!')

print('***prediction phase***')

model.load_state_dict(torch.load(FILENAME))
test_predictions = predict_pytorch(model, df_test)
print(test_predictions)



