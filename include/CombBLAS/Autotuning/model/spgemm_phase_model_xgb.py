import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau

import argparse
import statistics as stats
import os
import time
import random
import math
import json
import pickle

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
            "PPN",
            "rank"
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


class ProblemPhaseResults:


    def __init__(self):
        self.results = {}

    @dataclass
    class Result:
        problem:str
        rmse:float
        kt:float
        diff:float
        top1err:float
        correct:int

    def add_result(self, problem, y_arr, y_pred_arr):

        y_arr, y_pred_arr = np.array(y_arr), np.array(y_pred_arr)

        kt = (kendalltau(y_pred_arr, y_arr).statistic)

        rmse = ((((np.linalg.norm(y_pred_arr - y_arr))**(1/2))/len(y_arr))**2)

        is_correct = 1 if np.argmin(y_pred_arr)==np.argmin(y_arr) else 0

        diff = abs(y_arr[np.argmin(y_pred_arr)] - np.min(y_arr))

        if np.min(y_arr)>0:
            top_1_err = (y_arr[np.argmin(y_pred_arr)] / (np.min(y_arr))) - 1
        else:
            top_1_err = None

        self.results[problem] = self.Result(problem, rmse, kt, diff, top_1_err, is_correct)
    
    def get_stat_arr(self, stat_name):
        arr = map(lambda r: self.results[r].__dict__[stat_name], self.results)
        return list(filter(lambda x: x!=None, arr)) 

    def get_result_stat(self, problem, stat):
        return self.results[problem].__dict__[stat]

    def output_eval(self):

        kt_arr = self.get_stat_arr("kt")
        rmse_arr = self.get_stat_arr("rmse")
        diff_arr = self.get_stat_arr("diff")
        correct_arr = self.get_stat_arr("correct")
        err_arr = self.get_stat_arr("top1err")
        
        print(f"----AVERAGE KT: {sum(kt_arr)/len(kt_arr)}")
        print(f"----MEDIAN KT: {stats.median(kt_arr)}")
        
        kt_sorted_results = sorted(self.results.values(), key=lambda r:r.kt)
        print(f"----Problems with the 10 worst KT are: ")
        for i in range(0, 10):
            print(f"{kt_sorted_results[i].problem}")

        print(f"----Problems with the 10 best KT are: ")
        for i in range(1, 11):
            print(f"{kt_sorted_results[-i].problem}")

        print(f"----AVERAGE DIFF : {sum(diff_arr)/len(diff_arr)}s")
        print(f"----TOTAL DIFF : {sum(diff_arr)}s")
        print(f"----NUMBER CORRECT : {sum(correct_arr)}/{len(correct_arr)}")
        print(f"----TOTAL TOP 1 ERROR: {sum(err_arr)}")
        print(f"----AVERAGE TOP 1 ERROR: {sum(err_arr)/len(err_arr)}")
        print(f"----MEDIAN TOP 1 ERROR: {stats.median(err_arr)}")
    

def eval_phase(args, test_df, model, bulk=True):
    
    os.system(f"rm -f {args.label}-plots/*")

    test_df['params'] = test_df.apply(lambda row: f"{row['Nodes']}, {row['PPN']}", axis=1)
    test_df['processes'] = test_df.apply(lambda row: f"{row['Nodes']*row['PPN']}", axis=1)

    problems = test_df['problem'].unique()

    results = ProblemPhaseResults()
    i = 0
    
    print(f"Evaluating {len(problems)} problems")
    for problem in problems:

        if i%10==0:
            print(f"{i}/{len(problems)} evaluated...")

        df_problem = test_df[test_df['problem']==problem]
        
        params = df_problem['params'].unique()
        
        y_pred_arr = []
        y_arr = []
        processes = []
        valid_params = []

        if bulk:

            stime = time.time()

            X = df_problem[features]
            y = df_problem[args.label]
            y_pred = model(X)

            df_problem['pred'] = y_pred
            
            df_max = df_problem.groupby(by=["Nodes","PPN"]).max()
            y_pred_arr = df_max['pred']
            y_arr = df_max[args.label]

            valid_params = params

            etime = time.time()
            print(f"Time for inference: {etime-stime}s")
        else:
            for param in params:

                df_problem_params = df_problem[df_problem['params']==param]

                if not math.sqrt(df_problem_params.shape[0]).is_integer():
                    print("erroneous grid")
                    continue

                stime = time.time()
                X = df_problem_params[features]
                y = df_problem_params[args.label]
                y_pred = model(X)
                y_pred_arr.append(max(y_pred))
                y_arr.append(max(y))
                processes.append(df_problem_params['processes'].values[0])
                valid_params.append(param)
                etime = time.time()
                print(f"Time for inference: {etime-stime}s")
        

        results.add_result(problem, y_arr, y_pred_arr)

        if args.plotname:
            kt = results.get_result_stat(problem, "kt")
            rmse = results.get_result_stat(problem, "rmse")
            plt.scatter(valid_params, y_arr, label="Actual", marker='.',s=100)
            plt.scatter(valid_params, y_pred_arr, label=f"Predicted\n(kt={kt})\n(rmse={rmse})", marker='.',s=100)
            plt.ylabel("Runtime (s)")
            plt.yscale("log")
            plt.xlabel("Params")
            plt.xticks(rotation='vertical')
            plt.title(f"Actual vs. Predicted Runtime for {problem} {args.label}")
            plt.legend()
            plt.savefig(f"{args.label}-plots/{args.plotname}-{problem}.png", bbox_inches='tight')
            plt.clf()

        i+=1

    results.output_eval()


def split(df, size):    
    problems = df['problem'].unique()
    s = int(len(problems)*size)

    #random.shuffle(problems)
    #print(len(problems))

    test_problems, train_problems = problems[:s],problems[s:]
    
    test, train = df[df['problem'].isin(test_problems)], df[df['problem'].isin(train_problems)]
    return test,train 


def loc_mult_model(X, use_xgb=True):
    X = X.drop(labels=["rank"], axis=1)
    if use_xgb:
        bst = xgb.Booster()
        bst.load_model(f"{path_prefix}models/xgb-mult.model")
        return np.array(bst.predict(xgb.DMatrix(X)))
    else:
        return 0


def merge_model(X, use_xgb=True):
    X = X.drop(labels=["rank"], axis=1)
    if use_xgb:
        bst = xgb.Booster()
        bst.load_model(f"{path_prefix}models/xgb-merge.model")
        return np.array(bst.predict(xgb.DMatrix(X)))
    else:
        return 0

def bcast_model(X):

    def bcast_tree(grid_dim, msg_size):
        alpha = math.log2(grid_dim)*perlmutter_params.inter_alpha
        beta = math.log2(grid_dim)*(total_bytes / perlmutter_params.inter_beta)
        return (alpha+beta) / 1e6

    def bcast_ring(grid_dim, msg_size):
        alpha = (math.log2(grid_dim) + grid_dim - 1)*perlmutter_params.inter_alpha
        beta = ((2*(grid_dim-1))/grid_dim)*(msg_size/perlmutter_params.inter_beta)
        return (alpha+beta)/1e6

    def bcast_regression(nodes, ppn, msg_size):
        if math.isnan(msg_size):
            return 0
        with open("./bcast-bench/bcast-model.pkl", 'rb') as file:
            model = pickle.load(file)
            return model.predict(pd.DataFrame({"nodes":[nodes], "ppn":[ppn], "msg_size":[msg_size]}))[0]/(1e6)


    grid_dim = int(math.sqrt(X.shape[0]))

    times_mat = np.zeros(shape=(grid_dim, grid_dim))

    # Populate bytes
    for _, x in X.iterrows():

        ppn = x["PPN"]
        nodes = x["Nodes"]
        rank = int(x["rank"])

        nnz_A, nnz_B = x["nnz-A"], x["nnz-B"] 
        density_A = nnz_A/(x['n-A']*x['m-A'])
        density_B = nnz_B/(x['n-B']*x['m-B'])

        bytes_A = nnz_A * 8 + nnz_A * 8 +  nnz_A * 8
        bytes_B = nnz_B * 8 + nnz_B * 8 + nnz_B * 8
        
        row_rank = rank % grid_dim
        col_rank = rank // grid_dim

        total_bytes = bytes_A + bytes_B

        # Decide which bcast algorithm to use based on msg size
        #if total_bytes < 1e8 or True:
        bcast_time = bcast_tree(grid_dim, total_bytes)
        #else:
        #    bcast_time = bcast_ring(grid_dim, total_bytes)
        #bcast_time = bcast_regression(nodes, ppn, total_bytes)
        
        times_mat[row_rank, col_rank] = bcast_time

    row_times = np.sum(times_mat, axis=1)
    col_times = np.sum(times_mat, axis=0)

    for i in range(times_mat.shape[0]):
        for j in range(times_mat.shape[1]):
            times_mat[i,j] = row_times[i] + col_times[j]

    return times_mat.flatten()


def spgemm_model(X):
    bcast_pred = np.array([]) 
    for _, x in X.groupby(by=["Nodes","PPN"]):
        bcast_pred = np.append(bcast_pred, (bcast_model(x)))
    mult_pred = loc_mult_model(X, True)
    merge_pred = merge_model(X, True)
    
    #bcast_pred = np.zeros(shape=(X.shape[0]))

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
    parser.add_argument('--load', const=1, nargs='?', type=int)

    args = parser.parse_args()
    
    # Load in dataframe
    if args.load:
        df = load_gnn_df(features, labels) 

        # Only problems with 1,2,4 nodes
        all_problems = df['problem'].unique()
        valid_problems = [p for p in all_problems if df[df['problem']==p]['Nodes'].unique().shape[0]==3]
        df = df[df['problem'].isin(valid_problems)]
        
        # Make sure each grid is valid
        
        for p in valid_problems:
            df_problem = df[df['problem']==p]
            for _, d in df_problem.groupby(by=["Nodes", "PPN"]):
                if not math.sqrt(d.shape[0]).is_integer():
                    df = df.drop(labels=d.index)
            

        df.to_pickle("./master-df-gnn.pkl")
    else:
        df = pd.read_pickle("./master-df-gnn.pkl")
    
    df["no-bcast"] = df["local-mult"] + df["merge"]
    
    test_data, train_data = split(df, 0.1)
    
    if args.train:
        train_model_xgb(args, train_data, test_data)
    
    if args.eval=="mult":
        eval_phase(args, test_data, loc_mult_model)
    elif args.eval=="merge":
        eval_phase(args, test_data, merge_model)
    elif args.eval=="bcast":
        eval_phase(args,  test_data, bcast_model, False)
    elif args.eval=="spgemm":
        eval_phase(args,test_data, spgemm_model)
    

    
