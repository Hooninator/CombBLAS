
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid
from torch_geometric import nn

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

            for line in file:

                if "SAMPLE" in line: # new sample

                    if i>0:
                        if ill_formed:
                            bad_samples+=1
                            ill_formed = False
                        else:
                            (rows,cols),pos = grid(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])))
                            edge_index = torch.stack((rows, cols))
                            data = Data(x=torch.tensor(x), edge_index = edge_index, pos=pos, 
                                        y=torch.tensor(y))
                            data.validate(raise_on_error=True)
                            graph_list.append(data)
                    
                    x = np.zeros(shape=(0, n_features+1)) #One more for rank
                    y = np.zeros(shape=(0,1))

                    i+=1
                    
                else: # Extract line

                    feats = line.split(" ")
                    x_row = np.zeros(shape=(1, n_features+1))
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
                        x = np.append(x, x_row, axis=0) # Add row
                    except Exception as err:
                        ill_formed = True

            # Add last sample
            (rows,cols),pos = grid(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])))
            edge_index = torch.stack((rows, cols))
            data = Data(x=torch.tensor(x), edge_index = edge_index, pos=pos, 
                        y=torch.tensor(y))
            data.validate(raise_on_error=True)
            graph_list.append(data)
    
    etime = time.time()
    print(f"Processsed {len(graph_list)} graphs in {etime-stime}s")
    print(f"There were {bad_samples} ill-formed graphs")

    return graph_list 


def train(train_graphs):

    return


def eval(model, test_graphs):
    
    return


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", const=1, nargs='?', type=int)
    parser.add_argument("--eval", const=1, nargs='?', type=int)
    parser.add_argument("--label", type=str)

    args = parser.parse_args()
    
    graphs = make_graphs(args)

    random.shuffle(graphs)

    n_samples = len(graphs)
    test_size = int(0.10*n_samples)
    test_graphs, train_graphs = graphs[0:test_size], graphs[test_size:]

    model = train(train_graphs)








