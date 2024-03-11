
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

from scipy.stats import kendalltau

import argparse
import time
import os
import sys
import random
import math
import pickle

from hyperparams import HyperParams

path_prefix="/global/homes/j/jbellav/CombBLAS/include/CombBLAS/Autotuning/model/"
features = [
            "FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz-intermediate",
            "outputNnz-final"
            ]
n_features = len(features)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device")
else:
    device = torch.device('cpu')
    print("No CUDA device found, using CPU")

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
                        elif math.sqrt(x.shape[0])>0:
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
            if math.sqrt(x.shape[0])>0:
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

def write_graphs(graphs, fname="graphs.pkl"):
    return

class GNN(torch.nn.Module):
    def __init__(self, num_feats, embedding_size, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.conv1 = GCNConv(num_feats, embedding_size)
        self.conv2 = GCNConv(embedding_size, hidden_dim_1)
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



def train(args, params, train_loader, test_loader):
   
    model = GNN(n_features+1, int(params.embedding_size), int(params.hidden_size1), int(params.hidden_size2))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.alpha)
    criterion = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    for e in range(args.epochs):
        stime = time.time()
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            data.to(device)
            pred = model(data)
            loss = criterion(pred, data.y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0
        for data in test_loader:
            data.to(device)
            pred = model(data)
            loss = criterion(pred, data.y)
            test_loss += loss.item()
        
        test_losses.append(test_loss)

        etime = time.time()
        if e%10==0:
            print(f"-----------")
            print(f"Training loss at epoch {e}: {train_loss}")
            print(f"Test loss at epoch {e}: {test_loss}")
            print(f"Total time for epoch {e}: {etime-stime}s")
            print(f"-----------")


    return model, train_losses, test_losses


def eval_model(args, params, model, graphs, test_problems, train_problems):

    os.system("rm -f test-plots/* && rm -f train-plots/*")

    problems = test_problems + train_problems
    
    kt_sum_test = diff_sum_test = rmse_test = n_correct_test = 0
    kt_sum_train = diff_sum_train = rmse_train = n_correct_train = 0

    for problem in problems:

        print(f"Evaluating {problem}...")

        target_dir = "test-plots/" if problem in test_problems else "train-plots/"
        
        problem_graphs = filter(lambda g: g[1]==problem, graphs)
        problem_graphs = list(map(lambda g: g[0], problem_graphs))

        pred_arr = []
        y_arr = []
        for graph in problem_graphs:

            graph.to(device)

            pred = model(graph)
            y = graph.y

            pred_arr.append(pred.cpu().item())
            y_arr.append(y.cpu().item())

        kt = kendalltau(pred_arr, y_arr)
        diff = abs(y_arr[np.argmin(pred_arr)] - np.min(y_arr))
        rmse = ((np.linalg.norm(np.array(y_arr)-np.array(pred_arr))**2) / len(y_arr))**(1/2)
        correct = 1 if np.argmin(pred_arr)==np.argmin(y_arr) else 0

        if problem in test_problems:
            kt_sum_test += kt.correlation
            diff_sum_test += diff
            rmse_test += rmse
            n_correct_test += correct
        else:
            kt_sum_train += kt.correlation
            diff_sum_train += diff
            rmse_train += rmse
            n_correct_train += correct

        plt.scatter(range(len(y_arr)), y_arr, label="Actual", color='lime')
        plt.scatter(range(len(y_arr)), pred_arr, label="Model", color='navy')
        plt.ylabel("Runtime (s)")
        plt.xlabel("Sample ID")
        plt.title(f"Actual vs. Predicted Runtimes for {problem}")
        plt.legend()
        plt.savefig(f"{target_dir}{args.plotname}-{problem}.png", bbox_inches='tight')
        plt.clf()

    print("----TEST----")
    print(f"KT: {kt_sum_test}")
    print(f"DIFF: {diff_sum_test}s")
    print(f"RMSE: {rmse_test}")
    print(f"CORRECT: {n_correct_test}/{len(test_problems)}")
    print("----TRAIN----")
    print(f"KT: {kt_sum_train}")
    print(f"DIFF: {diff_sum_train}s")
    print(f"RMSE: {rmse_train}")
    print(f"CORRECT: {n_correct_train}/{len(train_problems)}")
    print("\n")


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

    #print(f"Test problems: {test_problems}")
    #print(f"Train problems: {train_problems}")

    return test_graphs, train_graphs, test_problems, train_problems


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", const=1, nargs='?', type=int)
    parser.add_argument("--eval", const=1, nargs='?', type=int)
    parser.add_argument("--label", type=str)
    parser.add_argument("--plotname", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--modelname", type=str)

    #parser.add_argument("--batch-size", type=int)
    #parser.add_argument("--alpha", type=float)
    #parser.add_argument("--embedding-size", type=int)
    #parser.add_argument("--hidden-size-1", type=int)
    #parser.add_argument("--hidden-size-2", type=int)
    parser.add_argument("--hparams", type=str)
    parser.add_argument("--randtest", const=1, nargs='?', type=int)
    
    args = parser.parse_args()
    
    params = HyperParams(args.hparams)

    graphs = make_graphs(args)

    if args.randtest != None:
        random.shuffle(graphs)

    test_graphs, train_graphs, test_problems, train_problems = split(graphs, 0.1) 
    
    train_loader = DataLoader(train_graphs, batch_size=int(params.batch_size), shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=int(params.batch_size), shuffle=True)

    if args.train != None:
        model, train_losses, test_losses = train(args, params, train_loader, test_loader)
        plot_losses(args, train_losses, test_losses)
        torch.save(model, f"./{args.modelname}")
    else:
        model = torch.load(args.modelname)

    if args.eval != None:
        eval_model(args, params, model, graphs, test_problems, train_problems)








