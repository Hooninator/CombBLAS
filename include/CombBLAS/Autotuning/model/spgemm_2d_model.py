# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: pytorch-2.0.1
#     language: python
#     name: pytorch-2.0.1
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
import argparse
import statistics as stats
import os
import time
import argparse
import random

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from collections import defaultdict

import pandas as pd

from scipy.stats import kendalltau

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import xgboost as xgb

from data_utils import *

path_prefix = "/global/homes/j/jbellav/CombBLAS/include/CombBLAS/Autotuning/model"
cores_per_node = 128

# List of feature names
feature_names = [ 
    "Nodes",
    "PPN",
    "avgDensityCol-A",
    "avgDensityCol-B",
    #"avgDensityTile-A",
    #"avgDensityTile-B",
    #"avgDensityTileCol-A",
    #"avgDensityTileCol-B",
    "avgNnzCol-A",
    "avgNnzCol-B",
    #"avgNnzTile-A",
    #"avgNnzTile-B",
    #"avgNnzTileCol-A",
    #"avgNnzTileCol-B",
    "density-A",
    "density-B",
    "m-A",
    "m-B",
    #"mTile-A",
    #"mTile-B",
    "maxDensityCol-A",
    "maxDensityCol-B",
    #"maxDensityTile-A",
    #"maxDensityTile-B",
    #"maxDensityTileCol-A",
    #"maxDensityTileCol-B",
    "maxNnzCol-A",
    "maxNnzCol-B",
    #"maxNnzTile-A",
    #"maxNnzTile-B",
    #"maxNnzTileCol-A",
    #"maxNnzTileCol-B",
    "minDensityCol-A",
    "minDensityCol-B",
    #"minDensityTile-A",
    #"minDensityTile-B",
    #"minDensityTileCol-A",
    #"minDensityTileCol-B",
    "minNnzCol-A",
    "minNnzCol-B",
    #"minNnzTile-A",
    #"minNnzTile-B",
    #"minNnzTileCol-A",
    #"minNnzTileCol-B",
    "n-A",
    "n-B",
    #"nTile-A",
    #"nTile-B",
    "nnz-A",
    "nnz-B",
    "stdevDensityCol-A",
    "stdevDensityCol-B",
    #"stdevDensityTile-A",
    #"stdevDensityTile-B",
    #"stdevDensityTileCol-A",
    #"stdevDensityTileCol-B",
    "stdevNnzCol-A",
    "stdevNnzCol-B",
    #"stdevNnzTile-A",
    #"stdevNnzTile-B",
    #"stdevNnzTileCol-A",
    #"stdevNnzTileCol-B"
]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.01)

class SpGEMMData(Dataset):
    def __init__(self, features, labels):
        self.features=features
        self.labels=labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.features[i, :], self.labels[i]


class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        hidden_size1 = 256
        hidden_size2 = 128
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )
        self.model.apply(init_weights)

    def forward(self, x):
        y = self.model(x)
        return y



# +
def train(dloader, model, loss_fn, optimizer, epoch, device):
    
    model.train()
    
    for _, (X, y) in enumerate(dloader):
        
        X, y = X.to(device), y.to(device)

        pred = model(X)
        
        loss = loss_fn(torch.flatten(pred), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()


def test(dloader, model, loss_fn, epoch, device):
    
    model.eval()
    
    with torch.no_grad():
        for _, (X, y) in enumerate(dloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(torch.flatten(pred), y)
    
    return loss.item()
                


def train_torch(args):
    
    if torch.cuda.is_available():
        print("CUDA found")
        device = torch.device("cuda")
        use_pinned=True
    else:
        print("CUDA not found, using CPU")
        device = torch.device("cpu")
        use_pinned=False


    # Load in dataframe
    df = load_spgemm2d_data(args.infile)

    X = df[feature_names]
    print(list(feature_names))

    # Label
    label_name = args.label
    y = df[label_name]

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    X_train, X_test, y_train, y_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)


    # Dataloaders
    batch_size=128

    training_data = SpGEMMData(X_train, y_train)
    test_data = SpGEMMData(X_test, y_test)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=use_pinned)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=use_pinned)

    # Make model
    model = LinearModel(len(feature_names))
    model.to(device)

    # Hyperparameters
    loss = nn.MSELoss()
    alpha = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=alpha)

    # -

    # Store losses for plotting 
    test_losses = []
    training_losses = []

    # +
    # Main loop
    best_loss = float('inf')
    epochs = 2500
    T = len(X_train)

    stime1 = time.time()
    for epoch in range(0, epochs+1):
        
        if epoch%1==0:
            print(f"Epoch: {epoch}")

        stime2 = time.time()

        for _ in range(T):
            training_loss = train(train_loader, model, loss, optimizer, epoch, device)
            
        if epoch%1==0:
            print(f"Training Loss: {training_loss}")
        
        test_loss = test(test_loader, model, loss, epoch, device)
        
        if epoch%1==0:
            print(f"Test Loss: {test_loss}")
        
        training_losses.append(training_loss)
        test_losses.append(test_loss)

        if test_loss<best_loss:
            torch.save(model, path_prefix+args.modelname+".pkl")
            best_loss = test_loss

        etime2 = time.time()
        
        if epoch%1==0:
            print(f"Time for epoch {epoch}: {etime2-stime2}")

    # -
    etime1 = time.time()

    print("Total training time: " + str(etime1-stime1))

    plt.plot(range(len(test_losses)), test_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Test Loss vs. Epochs")
    plt.savefig(args.plotname)

    
def infer_torch(args):
    
    model_path = path_prefix+"/models/"+args.modelname+".pkl"
    model = torch.load(model_path)
    model.to(torch.device("cpu")) # do inference on CPU
    
    df = load_spgemm2d_data(args.infile)
    
    X = torch.tensor(df[feature_names].values, dtype=torch.float32)
    y = torch.tensor(df[args.label].values, dtype=torch.float32)
    
    T = 100
    
    y_arr = []
    y_pred_arr = []
    for t in range(T):
        i = random.randint(0, y.shape[0]-1)

        X_i, y_i = X[i,:], y[i]

        y_pred = model(X_i)
        
        y_arr.append(y_i)
        y_pred_arr.append(y_pred.item())

        print(f"Predicted: {y_pred.item()}, actual: {y_i}")
    
    kendall = kendalltau(y_arr, y_pred_arr)
    
    plt.scatter(range(len(y_arr)), y_arr, label="Actual")
    plt.scatter(range(len(y_pred_arr)), y_pred_arr, label=f"Predicted\n(kt={kendall.correlation})")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Sample index")
    plt.title("Actual vs. Predicted Runtime for Local SpGEMM")
    plt.legend()
    plt.savefig(args.plotname)
    
    
# +


def train_model_xgb(args):
    
    # Load in dataframe
    df = load_spgemm2d_data(args.infile)

    X = df[feature_names]
    print(list(feature_names))

    # Label
    label_name = args.label
    y = df[label_name]

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    training_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)
    
    param = { 
        'max_depth':2,
        'eta':1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    evallist = [(training_data, 'train'), (test_data, 'eval')]
    
    nrounds = 100
    bst = xgb.train(param, training_data, nrounds, evallist)
    
    bst.save_model(f"{path_prefix}models/{args.modelname}.model")
    
    # Plot actual vs. predicted for test data    
    y_arr = y_test
    stime = time.time()
    y_pred_arr = bst.predict(test_data)
    etime = time.time()
    
    print(f"Time for inference on test dataset of size {y_test.shape[0]}: {etime-stime}s")
    
    kendall = kendalltau(y_arr, y_pred_arr)
    
    plt.scatter(range(len(y_arr)), y_arr, label="Actual")
    plt.scatter(range(len(y_pred_arr)), y_pred_arr, label=f"Predicted\n(kt={kendall.correlation})")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Sample index")
    plt.title(f"Actual vs. Predicted Runtime for {label_name}")
    plt.legend()
    plt.savefig(args.plotname)


def infer_xgb(args):
    
    # Load in dataframe
    df = load_spgemm2d_data(args.infile)

    X = np.array(df[feature_names].values)
    print(list(feature_names))

    # Label
    label_name = args.label
    y = np.array(df[label_name].values)
    
    # Model
    bst = xgb.Booster()
    bst.load_model(f"{path_prefix}models/{args.modelname}.model")
    
    y_arr = y
    y_pred_arr = bst.predict(xgb.DMatrix(X))

    kendall = kendalltau(y_arr, y_pred_arr)
    
    plt.scatter(range(len(y_arr)), y_arr, label="Actual")
    plt.scatter(range(len(y_pred_arr)), y_pred_arr, label=f"Predicted\n(kt={kendall.correlation})")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Sample index")
    plt.title("Actual vs. Predicted Runtime for 2D SpGEMM")
    plt.legend()
    plt.savefig(args.plotname)
    
    
    


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",type=str)
    parser.add_argument("--plotname",type=str)
    parser.add_argument("--label",type=str)
    parser.add_argument("--modelname",type=str)
    parser.add_argument("--train_torch", const=1, nargs='?', type=int)
    parser.add_argument("--infer_torch", const=1, nargs='?', type=int)
    parser.add_argument("--train_xgb", const=1, nargs='?', type=int)
    parser.add_argument("--infer_xgb", const=1, nargs='?', type=int)

    args = parser.parse_args()
    
    if args.train_torch:   
        train_model_torch(args)
    elif args.train_xgb:
        train_model_xgb(args)
    elif args.infer_torch:
        infer_torch(args)
    elif args.infer_xgb:
        infer_xgb(args)

    
