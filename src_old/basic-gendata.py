import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset

def window_generator(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
  
def generate_sin(look_back):
    n_features = 1

    print(train_series.shape)
    print(val_series.shape)
    print(test_series.shape)

    # Inspeção visual da série gerada
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(train_steps, train_series, lw=3, label='train data')
    ax.plot(val_steps, val_series, lw=3, label='val data')
    ax.plot(test_steps, test_series,  lw=3, label='test data')
    ax.legend(loc="lower left")
    plt.show();
    fig.savefig('sin.png', bbox_inches='tight')

    # Janelamento (data windowing)
    look_back = 4
    X_train, y_train =  window_generator(train_series, look_back)
    X_val, y_val =  window_generator(val_series, look_back)
    X_test, y_test =  window_generator(test_series, look_back)

    print(X_train.shape)
    print(X_train[:, 0].shape)
    print(y_train.shape)

    print(X_val.shape)
    print(y_val.shape)

    print(X_test.shape)
    print(y_test.shape)
  
    df_train = pd.DataFrame({'t4': X_train[:,0], 
                             't3': X_train[:,1], 
                             't2': X_train[:,2], 
                             't1': X_train[:,3], 
                             't0': y_train})
    df_train.to_csv('train_sin.csv', index=False)

    df_val = pd.DataFrame({'t4': X_val[:,0], 
                           't3': X_val[:,1], 
                           't2': X_val[:,2], 
                           't1': X_val[:,3], 
                           't0': y_val})
    df_val.to_csv('val_sin.csv', index=False)
    
    df_test = pd.DataFrame({'t4': X_test[:,0], 
                            't3': X_test[:,1], 
                            't2': X_test[:,2], 
                            't1': X_test[:,3], 
                            't0': y_test})
    df_test.to_csv('test_sin.csv', index=False)
    

train_steps = np.arange(0, 180, 1)
train_series = np.sin(train_steps)

val_steps = np.arange(176, 190, 1)
val_series = np.sin(val_steps)

test_steps = np.arange(186, 200, 1)
test_series = np.sin(test_steps)

look_back = 4

generate_sin(look_back)
