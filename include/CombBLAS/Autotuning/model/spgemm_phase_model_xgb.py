import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau

import argparse
import statistics as stats
import os
import time
import argparse
import random
import math
import json

from collections import defaultdict
from dataclasses import dataclass

import xgboost as xgb

from data_utils import *

path_prefix = "/global/homes/j/jbellav/CombBLAS/include/CombBLAS/Autotuning/model/"
cores_per_node = 128


features = ["FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz-intermediate",
            "outputNnz-final",
            "Nodes",
            "PPN"
            ]
labels = [
        "bcast-A",
        "bcast-B",
        "total-time",
        "local-mult",
        "summed-time",
        "merge"
]
n_features = len(features)


@dataclass
class PlatformParams:
    inter_beta: float
    inter_alpha: float

perlmutter_params = PlatformParams(2398.54, 3.9)


def load_gnn_df(f_prefix="samples-gnn-mod"):
    
    graph_list = []
    
    file_names = list(filter(lambda s: f_prefix in s, os.listdir(path_prefix)))
    
    df_dict = defaultdict(lambda: [])

    stime = time.time()
    for fname in file_names:
        print(f"Processing {fname}...")
        with open(path_prefix+fname, 'r') as file:
            for line in file:
                if "SAMPLE" in line:
                    continue
                feats = line.split(" ")
                found_feats = []
                for f in feats:
                    name, val = f.split(":")[0], f.split(":")[1]
                    name = name.strip()
                    if "problem" in name:
                        df_dict[name].append(val)
                        found_feats.append(name)
                    elif name in features+labels and name not in found_feats:
                        try:
                            df_dict[name].append(float(val))
                            found_feats.append(name)
                        except:
                            continue
                for f in features+labels+["problem"]:
                    if f not in found_feats:
                        df_dict[f].append(None)
    etime = time.time()
    df = pd.DataFrame(df_dict)
    print(f"Processsed {len(df)} samples in {etime-stime}s")

    return df 


def train_model_xgb(args, train_data, test_data):

    print(train_data.columns)
    X_train, X_test, y_train, y_test = train_data[features], test_data[features], train_data[args.label], test_data[args.label]

    training_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)
    
    param = { 
        'max_depth':2,
        'eta':1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    evallist = [(training_data, 'train'), (test_data, 'eval')]
    
    nrounds = args.epochs
    evals_result = {}
    bst = xgb.train(param, training_data, nrounds, evallist, verbose_eval=args.verbose, evals_result=evals_result)
    
    print(evals_result)

    bst.save_model(f"{path_prefix}models/{args.modelname}.model")
    
    plt.plot(range(nrounds), evals_result['train']['rmse'], label="train loss")
    plt.plot(range(nrounds), evals_result['eval']['rmse'], label="test loss")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title(f"Train and Test Loss")
    plt.legend()
    plt.savefig(f"{args.plotname}-loss", bbox_inches='tight')


def eval_xgb(args, model, df, features):
    
    
    os.system("rm -f plots-xgb")
    
    kendall_sum = 0
    n_kendall = 0
    rmse_sum = 0
    diff_sum = 0
    n_correct = 0
    top_1_err = 0
    n_top_1 = 0

    problems = df['problem'].unique()

    print(f"Evaluating {len(problems)} problems")

    for problem in problems:

        print(f"Evaluating {problem}")
        
        target_dir = "plots-xgb/"
        
        df_problem = df[df['problem']==problem].copy()
        X = df_problem[features]
        y = df_problem[args.label].values

        print(df_problem["Nodes"].unique())
        
        y_pred = model(X)

        df_problem['pred'] = y_pred

        df_pred = df_problem.groupby(['Nodes', 'PPN']).max()
        y_pred, y = np.array(df_pred['pred']), np.array(df_pred[args.label])
        
        is_correct = 1 if np.argmin(y_pred)==np.argmin(y) else 0
        diff = abs(y[np.argmin(y_pred)] - np.min(y))
        kendall = kendalltau(y, y_pred)
        rmse = ((np.linalg.norm(y - y_pred)**2)/len(y))**(1/2)
        
        
        if kendall!=float('nan'):
            kendall_sum += kendall.correlation
            n_kendall += 1

        rmse_sum += rmse
        diff_sum += diff
        n_correct += is_correct
        if np.min(y)>0:
            top_1_err += (y[np.argmin(y_pred)] / (np.min(y))) - 1
            n_top_1 += 1
            

        plt.scatter(range(len(y)), y, label="Actual")
        plt.scatter(range(len(y)), y_pred, label=f"Predicted\n(kt={kendall.correlation})\n(rmse={rmse})")
        plt.ylabel("Runtime (s)")
        plt.yscale("log")
        plt.xlabel("Sample index")
        plt.title(f"Actual vs. Predicted Runtime for {problem}")
        plt.legend()
        plt.savefig(f"{target_dir}{args.plotname}-{problem}.png")
        plt.clf()
    
    print(f"----AVERAGE KT : {kendall_sum/n_kendall}")
    print(f"----AVERAGE RMSE : {rmse_sum/len(problems)}")
    print(f"----AVERAGE DIFF : {diff_sum/len(problems)}s")
    print(f"----TOTAL DIFF : {diff_sum}s")
    print(f"----NUMBER CORRECT : {n_correct}/{len(problems)}")
    print(f"----TOTAL TOP 1 ERROR: {top_1_err}")
    print(f"----AVERAGE TOP 1 ERROR: {top_1_err/n_top_1}")
    

def split(df, size):    
    s = int(len(df)*size)
    test, train = df.iloc[0:s,:], df.iloc[s:,:]
    return test,train 


def loc_mult_model(X):
    bst = xgb.Booster()
    bst.load_model(f"{path_prefix}models/xgb-mult.model")

    return np.array(bst.predict(xgb.DMatrix(X)))


def merge_model(X):
    bst = xgb.Booster()
    bst.load_model(f"{path_prefix}models/xgb-merge.model")

    return np.array(bst.predict(xgb.DMatrix(X)))


def bcast_model(X):

    result_arr = []

    for _, x in X.iterrows():
        ppn = x.PPN
        nodes = x.Nodes

        total_procs = ppn * nodes

        nnz_A, nnz_B = x["nnz-A"], x["nnz-B"] 

        bytes_A = nnz_A * 8 + nnz_A * 8 + x['m-A'] * 8
        bytes_B = nnz_B * 8 + nnz_B * 8 + x['m-B'] * 8

        bcast_time_A = math.log2(total_procs)*perlmutter_params.inter_alpha + math.log2(total_procs)*(bytes_A / perlmutter_params.inter_beta)
        bcast_time_B = math.log2(total_procs)*perlmutter_params.inter_alpha + math.log2(total_procs)*(bytes_B / perlmutter_params.inter_beta)
        
        result_arr.append(bcast_time_A + bcast_time_B)
    
    return np.array(result_arr)


def spgemm_model(X):
    bcast_pred = bcast_model(X)
    mult_pred = loc_mult_model(X)
    merge_pred = merge_model(X)
    return bcast_pred + mult_pred + merge_pred


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--plotname",type=str)
    parser.add_argument("--label",type=str)
    parser.add_argument("--modelname",type=str)
    parser.add_argument("--train", const=1, nargs='?', type=int)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--verbose", const=1, nargs='?', type=int)
    parser.add_argument("--randtest", const=1, nargs='?', type=int)
    parser.add_argument("--epochs",  type=int)
    parser.add_argument("--nproblems", type=int)
    parser.add_argument("--neval", type=int)

    args = parser.parse_args()
    
    # Load in dataframe
    df = load_gnn_df() 
    
    test_data, train_data = split(df, 0.1)
    
    if args.train:
        train_model_xgb(args, train_data, test_data)
    
    if args.eval=="mult":
        eval_xgb(args, loc_mult_model, test_data, features)
    elif args.eval=="merge":
        eval_xgb(args, merge_model, test_data, features)
    elif args.eval=="bcast":
        eval_xgb(args, bcast_model, test_data, features)
    elif args.eval=="spgemm":
        eval_xgb(args, spgemm_model, df, features)
    

    
