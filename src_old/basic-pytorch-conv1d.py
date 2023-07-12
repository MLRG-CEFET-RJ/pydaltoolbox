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

# https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
import functools
import operator

class Conv1DNet(nn.Module):
    def __init__(self, in_channels, input_dim, kernel_size = 2):
        super(Conv1DNet,self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size),
            nn.ReLU(inplace=True)
        )
        
        # https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch
        num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(1, input_dim)).shape))
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=num_features_before_fcnn, out_features=50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self,x):
        out = self.feature_extractor(x)
        out = nn.Flatten(1, -1)(out)
        out = self.regressor(out)
        return out
      
def savemodel(model, filename):
  torch.save(model, filename)
  

def loadmodel(filename):
  model = torch.load(filename)
  return(model)


def savedf(data, filename):      
    data.to_csv(filename, index=False)
    
    
def create_model(in_channels, input_dim, kernel_size):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = Conv1DNet(in_channels, input_dim, kernel_size).to(device)
  return(model)
    
def train_pytorch(model, df_train, df_val, filename, n_epochs = 3000):
	X_train = df_train.drop('t0', axis=1).to_numpy()
	y_train = df_train.t0.to_numpy()

	X_train = X_train[:, :, np.newaxis]
	y_train = y_train[:, np.newaxis]
	
	train_x = torch.from_numpy(X_train)
	train_y = torch.from_numpy(y_train)
	
	train_x = torch.permute(train_x, (0, 2, 1))

	print(train_x.shape)
	
	train_ds = TensorDataset(train_x, train_y)
	
	X_val = df_val.drop('t0', axis=1).to_numpy()
	y_val = df_val.t0.to_numpy()
	
	X_val = X_val[:, :, np.newaxis]
	y_val = y_val[:, np.newaxis]
	
	valid_x = torch.from_numpy(X_val)
	valid_x = torch.permute(valid_x, (0, 2, 1))

	valid_y = torch.from_numpy(y_val)
	
	valid_ds = TensorDataset(valid_x, valid_y)
	
	BATCH_SIZE = 8
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
	valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle = True)
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
	fig.savefig('loss_plot-conv1d.png', bbox_inches='tight')
	return model


def predict_pytorch(model, df_test):
  X_test = df_test.drop('t0', axis=1).to_numpy()
  y_test = df_test.t0.to_numpy()
  
  X_test = X_test[:, :, np.newaxis]
  y_test = y_test[:, np.newaxis]
  
  test_x = torch.from_numpy(X_test)
  test_y = torch.from_numpy(y_test)
  
  test_x = torch.permute(test_x, (0, 2, 1))	

  print('test_x.shape: ', test_x.shape)
  
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
INPUT_DIM = 4
KERNEL_SIZE = 2

# df_train = pd.read_csv('./data/train_pesticide1.csv')
# df_val = pd.read_csv('./data/val_pesticide1.csv')
# df_test = pd.read_csv('./data/test_pesticide1.csv')
# INPUT_SIZE = 1

# df_train = pd.read_csv('./data/train_pesticide2.csv')
# df_val = pd.read_csv('./data/val_pesticide2.csv')
# df_test = pd.read_csv('./data/test_pesticide2.csv')
# INPUT_SIZE = 2

print('There are %d training examples' % df_train.shape[0])
print('There are %d validation examples' % df_val.shape[0])
print('There are %d test examples' % df_test.shape[0])

TRAIN_MODE = True
FILENAME = 'conv1d.model'

if TRAIN_MODE:
  print('Creating the model...')
  model = create_model(in_channels = 1, input_dim = INPUT_DIM, kernel_size = KERNEL_SIZE)
  print(model)
  print('Done!')
  print('Training the model...')
  train_pytorch(model, df_train, df_val, FILENAME, n_epochs = 500)
  print('Done!')

print('***prediction phase***')

model.load_state_dict(torch.load(FILENAME))
test_predictions = predict_pytorch(model, df_test)
print(test_predictions)



