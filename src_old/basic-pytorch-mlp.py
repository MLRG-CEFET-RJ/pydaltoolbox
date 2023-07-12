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

class MLPNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MLPNet,self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
      
def savemodel(model, filename):
  torch.save(model, filename)
  

def loadmodel(filename):
  model = torch.load(filename)
  model.eval()
  return(model)


def savedf(data, filename):      
    data.to_csv(filename, index=False)
    
    
def create_model(input_size, hidden_size):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = MLPNet(in_size=int(input_size), hidden_size=int(hidden_size), out_size=1).to(device)
  return model
    
def train_pytorch(model, df_train, df_val, filename, n_epochs = 3000):
    X_train = df_train.drop('t0', axis=1).to_numpy()
    y_train = df_train.t0.to_numpy()
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_train, y_train = X_train.float(), y_train.float()
    train_ds = TensorDataset(X_train, y_train)
    
    X_val = df_val.drop('t0', axis=1).to_numpy()
    y_val = df_val.t0.to_numpy()
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)
    X_val, y_val = X_val.float(), y_val.float()
    valid_ds = TensorDataset(X_val, y_val)
    
    BATCH_SIZE = 8
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle = False)
    
    PATIENCE = 20
    model = model.float()
    criterion = nn.MSELoss()
    
    train_loss, valid_loss = fit(model, train_loader, valid_loader, n_epochs, lr = 1e-5, criterion = criterion, patience = PATIENCE, filename = filename, opt_func=torch.optim.Adam)
    
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
    fig.savefig('loss_plot_mlp.png', bbox_inches='tight')
    return model


def predict_pytorch(model, df_test):
  X_test = df_test.drop('t0', axis=1).to_numpy()
  y_test = df_test.t0.to_numpy()
  
  X_test = X_test[:, :, np.newaxis]
  y_test = y_test[:, np.newaxis]
  
  test_x = torch.from_numpy(X_test)
  test_y = torch.from_numpy(y_test)
  
  # remove a  terceira (última) dimensão do tensor
  # e.g., torch.Size([10, 4, 1]) --> torch.Size([10, 4])
  test_x = torch.squeeze(test_x, 2)

  test_ds = TensorDataset(test_x, test_y)
  
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
INPUT_SIZE = 4

#df_train = pd.read_csv('./data/train_pesticide1.csv')
#df_val = pd.read_csv('./data/val_pesticide1.csv')
#df_test = pd.read_csv('./data/test_pesticide1.csv')
#INPUT_SIZE = 1

print('There are %d training examples' % df_train.shape[0])
print('There are %d validation examples' % df_val.shape[0])
print('There are %d test examples' % df_test.shape[0])

TRAIN_MODE = True
FILENAME = 'mlp.model'

if TRAIN_MODE:
  print('Creating the model...')
  model = create_model(input_size = INPUT_SIZE, hidden_size = 3)
  print(model)
  print('Done!')
  print('Training the model...')
  train_pytorch(model, df_train, df_val, FILENAME, n_epochs = 3000)
  print('Done!')

model.load_state_dict(torch.load(FILENAME))
test_predictions = predict_pytorch(model, df_test)
print(test_predictions)



