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
import math

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

path_prefix = "/global/homes/j/jbellav/CombBLAS/include/CombBLAS/Autotuning/model/"
cores_per_node = 128

# List of feature names
feature_names = [ 
    "Nodes",
    "PPN",
    "avgDensityCol-A",
    "avgDensityCol-B",
    "avgDensityTile-A",
    "avgDensityTile-B",
    "avgDensityTileCol-A",
    "avgDensityTileCol-B",
    "avgNnzCol-A",
    "avgNnzCol-B",
    "avgNnzTile-A",
    "avgNnzTile-B",
    "avgNnzTileCol-A",
    "avgNnzTileCol-B",
    "density-A",
    "density-B",
    "m-A",
    "m-B",
    "mTile-A",
    "mTile-B",
    "maxDensityCol-A",
    "maxDensityCol-B",
    "maxDensityTile-A",
    "maxDensityTile-B",
    "maxDensityTileCol-A",
    "maxDensityTileCol-B",
    "maxNnzCol-A",
    "maxNnzCol-B",
    "maxNnzTile-A",
    "maxNnzTile-B",
    "maxNnzTileCol-A",
    "maxNnzTileCol-B",
    "minDensityCol-A",
    "minDensityCol-B",
    "minDensityTile-A",
    "minDensityTile-B",
    "minDensityTileCol-A",
    "minDensityTileCol-B",
    "minNnzCol-A",
    "minNnzCol-B",
    "minNnzTile-A",
    "minNnzTile-B",
    "minNnzTileCol-A",
    "minNnzTileCol-B",
    "n-A",
    "n-B",
    "nTile-A",
    "nTile-B",
    "nnz-A",
    "nnz-B",
    "stdevDensityCol-A",
    "stdevDensityCol-B",
    "stdevDensityTile-A",
    "stdevDensityTile-B",
    "stdevDensityTileCol-A",
    "stdevDensityTileCol-B",
    "stdevNnzCol-A",
    "stdevNnzCol-B",
    "stdevNnzTile-A",
    "stdevNnzTile-B",
    "stdevNnzTileCol-A",
    "stdevNnzTileCol-B"
]


def train_model_xgb(args, X_train, y_train, X_test, y_test):
    
    training_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)
    
    param = { 
        'max_depth':2,
        'eta':1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    
    evallist = [(training_data, 'train'), (test_data, 'eval')]
    
    nrounds = 1000
    bst = xgb.train(param, training_data, nrounds, evallist)
    
    bst.save_model(f"{path_prefix}models/{args.modelname}.model")


def eval_xgb(args, df, feature_names, label, test_problems, train_problems):
    
    # Model
    bst = xgb.Booster()
    bst.load_model(f"{path_prefix}models/{args.modelname}.model")
    
    os.system("rm -f test-plots/* && rm -f train-plots/*")
    
    kendall_sum_train = 0
    kendall_sum_test = 0
    rmse_sum_train = 0
    rmse_sum_test = 0
    diff_sum_train = 0
    diff_sum_test = 0
    n_correct_train = 0
    n_correct_test = 0
    
    for problem in test_problems+train_problems:
        
        target_dir = "test-plots/" if problem in test_problems else "train-plots/"
        
        df_problem = df[df['trial']==problem]
        
        best_time_actual = df_problem['summed-time'].min(axis=0)
        
        X = df_problem[feature_names].values
        y = df_problem[[label]].values
        
        y_pred = bst.predict(xgb.DMatrix(X))
        
        best_time_predicted = df_problem['summed-time'].iloc[np.argmin(y_pred)]
        
        is_correct = int(np.argmin(np.array(df_problem['summed-time']))==np.argmin(y_pred))
        
        diff = abs(best_time_actual - best_time_predicted)
        
        kendall = kendalltau(y, y_pred)
        rmse = ((np.linalg.norm(y - y_pred)**2)/len(y))**(1/2)
        
        if problem in test_problems:
            kendall_sum_test += kendall.correlation
            rmse_sum_test += rmse
            diff_sum_test += diff
            n_correct_test += is_correct
        else:
            kendall_sum_train += kendall.correlation
            rmse_sum_train += rmse
            diff_sum_train += diff
            n_correct_train += is_correct

        plt.scatter(range(len(y)), y, label="Actual")
        plt.scatter(range(len(y)), y_pred, label=f"Predicted\n(kt={kendall.correlation})\n(rmse={rmse})")
        plt.ylabel("Runtime (s)")
        plt.xlabel("Sample index")
        plt.title(f"Actual vs. Predicted Runtime for {problem}")
        plt.legend()
        plt.savefig(f"{target_dir}{args.plotname}-{problem}.png")
        plt.clf()
    
    print(f"----AVERAGE KT FOR TRAINING DATA: {kendall_sum_train/len(train_problems)}")
    print(f"----AVERAGE KT FOR TEST DATA: {kendall_sum_test/len(test_problems)}")
    print(f"----AVERAGE RMSE FOR TRAINING DATA: {rmse_sum_train/len(train_problems)}")
    print(f"----AVERAGE RMSE FOR TEST DATA: {rmse_sum_test/len(test_problems)}")
    print(f"----AVERAGE DIFF FOR TRAINING DATA: {diff_sum_train/len(train_problems)}s")
    print(f"----AVERAGE DIFF FOR TEST DATA: {diff_sum_test/len(test_problems)}s")
    print(f"----NUMBER CORRECT FOR TRAINING DATA: {n_correct_train}/{len(train_problems)}")
    print(f"----NUMBER CORRECT FOR TEST DATA: {n_correct_test}/{len(test_problems)}")
    

def split(df, feature_names, label, size):

    df['trial'] = df.apply(lambda row: f"{row['A-name']}x{row['B-name']}", axis=1)
    
    problem_list = list(df['trial'].unique())
    random.shuffle(problem_list)
    
    s = math.floor(len(problem_list)*size)
    test_problems, train_problems = problem_list[0:s], problem_list[s:]
    
    df_test, df_train = df[df['trial'].isin(test_problems)], df[df['trial'].isin(train_problems)]
    
    X_test, y_test, X_train, y_train = df_test[feature_names], df_test[[label]], df_train[feature_names], df_train[[label]]
    
    return X_train, X_test, y_train, y_test, train_problems, test_problems


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",type=str)
    parser.add_argument("--plotname",type=str)
    parser.add_argument("--label",type=str)
    parser.add_argument("--modelname",type=str)
    parser.add_argument("--train_xgb", const=1, nargs='?', type=int)
    parser.add_argument("--eval_xgb", const=1, nargs='?', type=int)

    args = parser.parse_args()
    
    # Load in dataframe
    df = load_spgemm2d_data(args.infile)
    
    X_train, X_test, y_train, y_test, train_problems, test_problems = split(df, feature_names, args.label, 0.33)
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    if args.train_xgb:
        train_model_xgb(args, X_train, y_train, X_test, y_test)
    if args.eval_xgb:
        print(f"----Test problems----\n{test_problems}")
        print(f"----Train problems----\n{train_problems}")
        eval_xgb(args, df, feature_names, args.label, test_problems, train_problems)

    
