
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import grid
from torch_geometric.nn import GCNConv, global_mean_pool 
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU, Dropout

import argparse
import time
import os
import sys
import random
import math

path_prefix="/global/homes/j/jbellav/CombBLAS/include/CombBLAS/Autotuning/model/"
features = [
            "FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz"
            ]
n_features = len(features)

def make_graphs(args, f_prefix="samples-gnn"):

    graph_list = []
    file_names = list(filter(lambda s: f_prefix in s, os.listdir(path_prefix)))
    bad_samples = 0

    stime = time.time()
    for fname in file_names:

        print(f"Processing {fname}...")

        with open(path_prefix+fname, 'r') as file:

            ill_formed = False
            i = 0
            problem_name = ""

            for line in file:

                if "SAMPLE" in line: # new sample

                    if i>0:
                        if ill_formed:
                            bad_samples+=1
                            ill_formed = False
                        else:
                            (rows,cols),pos = grid(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])))
                            edge_index = torch.stack((rows, cols))
                            data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index = edge_index, pos=pos, 
                                        y=torch.tensor(y, dtype=torch.float32))
                            data.validate(raise_on_error=True)
                            graph_list.append((data, problem_name))
                    
                    x = np.zeros(shape=(0, n_features+1)) #One more for rank
                    y = np.zeros(shape=(0,1))

                    i+=1

                    problem_name = ""
                    
                else: # Extract line

                    feats = line.split(" ")
                    x_row = np.zeros(shape=(1, n_features+1))
                    problem_name_tmp = ""
                    try:
                        for f in feats:
                            if f=="\n":
                                continue
                            name, val = f.split(":")[0], (f.split(":")[1])
                            if name in features:
                                j = features.index(name)
                                x_row[0,j] = np.float32(val)
                            elif name==args.label:
                                y = np.append(y, np.array([[np.float32(val)]]), axis=0)
                            elif "name" in name:
                                problem_name_tmp+=val.split("/")[-1].split(".")[0]
                        x = np.append(x, x_row, axis=0) # Add row
                        if problem_name_tmp!="":
                            problem_name = problem_name_tmp
                    except Exception as err:
                        ill_formed = True

            # Add last sample
            (rows,cols),pos = grid(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])))
            edge_index = torch.stack((rows, cols))
            data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index = edge_index, pos=pos, 
                        y=torch.tensor(y, dtype=torch.float32))
            data.validate(raise_on_error=True)
            graph_list.append((data, problem_name))
    
    etime = time.time()
    print(f"Processsed {len(graph_list)} graphs in {etime-stime}s")
    print(f"There were {bad_samples} ill-formed graphs")

    return graph_list 


class GNN(torch.nn.Module):
    def __init__(self, num_feats, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.conv1 = GCNConv(num_feats, hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_1)
        self.dropout = Dropout(0.5)
        self.pool = global_mean_pool
        self.fc1 = Linear(hidden_dim_1, hidden_dim_2)
        self.fc2 = Linear(hidden_dim_2,1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First convolutional layer
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x) 
        
        # Second convolutional layer
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)

        # First linear layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)
        return x



def train(train_loader, test_loader):
   
    model = GNN(n_features+1, 256, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    epochs = 100

    train_losses = []
    test_losses = []
    for e in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data.y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Training loss at epoch {e}: {train_loss}")
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0
        for data in test_loader:
            pred = model(data)
            loss = criterion(pred, data.y)
            test_loss += loss.item()
        
        print(f"Test loss at epoch {e}: {test_loss}")
        test_losses.append(test_loss)


    return model, train_losses, test_losses


def eval(model, test_loader):
    
    return


def plot_losses(args, train_losses, test_losses):
    plt.plot(range(len(train_losses)), train_losses, label="train")
    plt.plot(range(len(test_losses)), test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("Test and Training Loss")
    plt.legend()
    plt.savefig(args.plotname)


def split(graphs, test_size=0.10):
    problems = list(set(map(lambda g: g[1], graphs)))
    n_problems = len(problems)

    test_size = int(n_problems*test_size)
    test_problems, train_problems = problems[0:test_size], problems[test_size:]

    test_graphs, train_graphs = [], []

    for g in graphs:
        if g[1] in test_problems:
            test_graphs.append(g[0])
        else:
            train_graphs.append(g[0])
    
    print(f"Train size: {n_problems-test_size}")
    print(f"Test size: {test_size}")

    print(f"Test problems: {test_problems}")
    print(f"Train problems: {train_problems}")

    return test_graphs, train_graphs, test_problems, train_problems


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", const=1, nargs='?', type=int)
    parser.add_argument("--eval", const=1, nargs='?', type=int)
    parser.add_argument("--label", type=str)
    parser.add_argument("--plotname", type=str)

    args = parser.parse_args()
    
    graphs = make_graphs(args)

    random.shuffle(graphs)

    n_samples = len(graphs)
    test_size = int(0.10*n_samples)
    test_graphs, train_graphs, test_problems, train_problems = split(graphs, 0.3) 
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=True)

    model, train_losses, test_losses = train(train_loader, test_loader)
    plot_losses(args, train_losses, test_losses)








