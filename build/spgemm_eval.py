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
import subprocess

from collections import defaultdict
from dataclasses import dataclass

import xgboost as xgb

from data_utils import *

path_prefix = "/global/homes/j/jbellav/CombBLAS/build/"
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
    gamma:float

perlmutter_params = PlatformParams(23980.54, 3.9, 5.2e-9)


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
        top2err:float
        top3err:float

        correct1:int
        correct2:int
        correct3:int

        spgemm_runtime:float
        timings:dict

    def add_result(self, problem, y_arr, y_pred_arr, spgemm_runtime, timings):

        y_arr, y_pred_arr = np.array(y_arr), np.array(y_pred_arr)

        kt = (kendalltau(y_pred_arr, y_arr).statistic)

        rmse = ((((np.linalg.norm(y_pred_arr - y_arr))**(1/2))/len(y_arr))**2)

        is_correct1 = 1 if np.argmin(y_pred_arr)==np.argmin(y_arr) else 0
        is_correct2 = 1 if is_correct1 or np.argpartition(y_pred_arr,1)[1]==np.argmin(y_arr) else 0
        is_correct3 = 1 if is_correct2 or np.argpartition(y_pred_arr,2)[2]==np.argmin(y_arr) else 0

        diff = abs(y_arr[np.argmin(y_pred_arr)] - np.min(y_arr))

        if np.min(y_arr)>0:
            top_1_err = (y_arr[np.argmin(y_pred_arr)] / (np.min(y_arr))) - 1
            top_2_err = min(top_1_err, (y_arr[np.argpartition(y_pred_arr, 1)[1]] / (np.min(y_arr))) - 1) 
            top_3_err = min(top_2_err, (y_arr[np.argpartition(y_pred_arr, 2)[2]] / (np.min(y_arr))) - 1) 
        else:
            top_1_err = None
            top_2_err = None
            top_3_err = None

        timings["AutotuningSpGEMM"] = y_arr[np.argmin(y_pred_arr)]
        self.results[problem] = self.Result(problem, rmse, kt, diff, top_1_err, top_2_err, top_3_err, 
                                            is_correct1, is_correct2, is_correct3, spgemm_runtime, timings)
    
    def get_stat_arr(self, stat_name):
        arr = map(lambda r: self.results[r].__dict__[stat_name], self.results)
        return list(filter(lambda x: x!=None, arr)) 

    def get_result_stat(self, problem, stat):
        return self.results[problem].__dict__[stat]

    def output_eval(self):

        kt_arr = self.get_stat_arr("kt")
        rmse_arr = self.get_stat_arr("rmse")
        diff_arr = self.get_stat_arr("diff")
        correct_arr1 = self.get_stat_arr("correct1")
        correct_arr2 = self.get_stat_arr("correct2")
        correct_arr3 = self.get_stat_arr("correct3")
        err_arr1 = self.get_stat_arr("top1err")
        err_arr2 = self.get_stat_arr("top2err")
        err_arr3 = self.get_stat_arr("top3err")
        
        print(f"----AVERAGE KT: {sum(kt_arr)/len(kt_arr)}")
        print(f"----MEDIAN KT: {stats.median(kt_arr)}")
        
        kt_sorted_results = sorted(self.results.values(), key=lambda r:r.kt)
        print(f"----Problems with the 10 worst KT are: ")
        for i in range(0, min(10, len(kt_sorted_results))):
            print(f"{kt_sorted_results[i].problem}")

        print(f"----Problems with the 10 best KT are: ")
        for i in range(1, min(11, len(kt_sorted_results)+1)):
            print(f"{kt_sorted_results[-i].problem}")

        print(f"----AVERAGE DIFF : {sum(diff_arr)/len(diff_arr)}s")
        print(f"----TOTAL DIFF : {sum(diff_arr)}s")
        print(f"----NUMBER CORRECT1 : {sum(correct_arr1)}/{len(correct_arr1)}")
        print(f"----NUMBER CORRECT2 : {sum(correct_arr2)}/{len(correct_arr2)}")
        print(f"----NUMBER CORRECT3 : {sum(correct_arr3)}/{len(correct_arr3)}")
        print(f"----AVERAGE TOP 1 ERROR: {sum(err_arr1)/len(err_arr1)}")
        print(f"----MEDIAN TOP 1 ERROR: {stats.median(err_arr1)}")
        print(f"----AVERAGE TOP 2 ERROR: {sum(err_arr2)/len(err_arr2)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr2)}")
        print(f"----AVERAGE TOP 3 ERROR: {sum(err_arr3)/len(err_arr3)}")
        print(f"----MEDIAN TOP 3 ERROR: {stats.median(err_arr3)}")

    def plot_eval(self):
        
        problems = []
        err_arr1 = []
        err_arr2 = []
        err_arr3 = []
        for problem in self.results.keys():
            err1 = self.get_result_stat(problem, "top1err")
            err2 = self.get_result_stat(problem, "top2err")
            err3 = self.get_result_stat(problem, "top3err")
            
            if err1 or err2 or err3:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                err_arr1.append(err1)
                err_arr2.append(err2)
                err_arr3.append(err3)

        width=2.0
        inds = np.arange(len(problems))*(width*4)
        plt.figure(figsize=(12,6))
        plt.bar(inds, err_arr1, label="Top 1 Error", width=width)
        plt.bar(inds+width, err_arr2, label="Top 2 Error", width=width)
        plt.bar(inds+width*2, err_arr3, label="Top 3 Error", width=width)
        plt.legend()
        plt.xticks(inds+width, labels=problems, rotation=90)
        plt.ylabel("Error")
        plt.title("Top K Errors of Test Matrices")
        plt.savefig(f"{args.label}-plots/{args.plotname}-errs.png", bbox_inches='tight')
        plt.clf()

    def plot_spgemm(self):

        problems = []
        spgemm_times = []
        autotuning_timings = []
        autotuning_spgemm_timings = []
        for problem in self.results.keys():
            spgemm_time = self.get_result_stat(problem, "spgemm_runtime")
            if spgemm_time:
                problems.append(problem.split(".")[0] + ("-permuted" if "permuted" in problem else ""))
                spgemm_times.append(spgemm_time)
                autotuning_timings.append(self.get_result_stat(problem, "timings"))
        
        feature_init_times = list(map(lambda t: t["FeatureInit"], autotuning_timings))
        prediction_times = list(map(lambda t: t["Prediction"], autotuning_timings))
        autotuning_spgemm_times = list(map(lambda t: t["AutotuningSpGEMM"], autotuning_timings))
        tuning_times = list(map(lambda t: t["TuneSpGEMM2D"], autotuning_timings))

        categories = ["Autotuning Runtime", "SpGEMM Runtime"]
        ind = np.arange(len(problems))*1.5
        plt.figure(figsize=(12,6))
        fig, ax = plt.subplots()
        #ax.bar(ind, feature_init_times, width=0.5, label="FeatureInit")
        #ax.bar(ind, prediction_times, width=0.5, label="Prediction", bottom=feature_init_times)
        ax.bar(ind, tuning_times, width=0.5, label="Autotuning Overhead")
        ax.bar(ind, autotuning_spgemm_times, width=0.5, label="Autotuning SpGEMM", bottom=tuning_times)
        ax.bar(ind+0.5, spgemm_times, width=0.5, label="Naive SpGEMM")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"SpGEMM Runtime vs. Autotuning Overhead")
        ax.set_xticks(ind)
        ax.set_xticklabels(problems, rotation=90)
        ax.legend()
        plt.savefig(f"{args.label}-plots/timing.png", bbox_inches='tight')
        plt.clf()


def eval_spgemm(args, test_df):
    
    test_df['params'] = test_df.apply(lambda row: f"{row['Nodes']}, {row['PPN']}", axis=1)
    test_df['processes'] = test_df.apply(lambda row: f"{row['Nodes']*row['PPN']}", axis=1)

    problems = test_df['problem'].unique()

    if args.problem:
        problems = [f"{args.problem}.mtx{args.problem}.mtx\n"]

    results = ProblemPhaseResults()
    i = 0
    
    print(f"Evaluating {len(problems)} problems")
    for problem in problems:

        if i%10==0:
            print(f"{i}/{len(problems)} evaluated...")

        df_problem = test_df[test_df['problem']==problem]
        
        params = df_problem['params'].unique()
        
        y_pred_arr = np.zeros(shape=(len(params)))
        y_arr = []
        processes = []
        valid_params = []


        X = df_problem[features]
        y = df_problem[args.label]
        
        nodes_cmd = int(df_problem["Nodes"].max())
        ppn_cmd = 64 if math.sqrt(nodes_cmd).is_integer() else 128
        permuted = 1 if "permuted" in problem else 0
        threads = 4 if ppn_cmd==64 else 2

        mat_name = problem.split(".")[0]
        
        cmd = f"export OMP_NUM_THREADS={threads} && srun --tasks-per-node {ppn_cmd} -N {nodes_cmd} Applications/autotune /pscratch/sd/j/jbellav/matrices/{mat_name}/{mat_name}.mtx /pscratch/sd/j/jbellav/matrices/{mat_name}/{mat_name}.mtx {permuted} {args.method}"

        print(f"Executing {cmd}...")


        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            result.check_returncode()
        except:
            print(result.stderr)
            os.system(f"rm -f info-{mat_name}x{mat_name}*")
            os.system("rm -f logfile*")
            i+=1
            continue
        
        timings = {}
        with open(f"info-{mat_name}x{mat_name}-0.out", 'r') as file:
            for line in file:
                if line.find("RUNTIME ESTIMATES")!=-1:
                    
                    line = next(file)
                    trials = line.split(" ")
                    
                    for trial in trials[:-1]:
                        params_curr,runtime = trial.split(":")[0], float(trial.split(":")[1][:-1])
                        nodes,ppn = float(params_curr.split(",")[0]), float(params_curr.split(",")[1])
                        if f"{nodes}, {ppn}" in params:
                            y_pred_arr[list(params).index(f"{nodes}, {ppn}")] = runtime
                if line.find("Prediction:")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["Prediction"] = t
                if line.find("FeatureInit:")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["FeatureInit"] = t
                if line.find("TuneSpGEMM2D")!=-1 and line.find("%")==-1:
                    t = float(line.split(":")[1])
                    timings["TuneSpGEMM2D"] = t


        os.system(f"rm -f info-{mat_name}x{mat_name}*")
        os.system("rm -f logfile*")

        y_arr = np.zeros(shape=(len(params)))
        
        for param in params:
            param_time = df_problem[df_problem["params"]==param][args.label].max()
            y_arr[list(params).index(param)] = param_time 


        spgemm_runtime = float(result.stdout.split("[Total]:")[1]) 
        results.add_result(problem, y_arr, y_pred_arr, spgemm_runtime, timings)

        i+=1

    with open(f"cpp-results-{args.method}-nonnz.pkl", 'wb') as picklefile:
        pickle.dump(results, picklefile)


def correctness(df, mat_name):

    problem = f"{mat_name}.mtx{mat_name}.mtx\n"

    df_problem = df[df["problem"]==problem]

    df_problem = df_problem.groupby(by=["Nodes","PPN"])
    
    logfiles = list(filter(lambda s: "logfile" in s, os.listdir(".")))

    n_features = 11 

    for (nodes, ppn), df_params in df_problem:
        
        for logfile in logfiles:

            with open(logfile, 'r') as file:
                for line in file: 
                    if "FeatureMat" in line:
                        features_data = line.split(":")[1].split(" ")

                        ranks = len(features_data)//n_features

                        file_nodes = float(features_data[9])
                        file_ppn = float(features_data[10])

                        print(nodes, file_nodes, ppn, file_ppn)

                        if nodes!=file_nodes or ppn!=file_ppn:
                            break

                        correct = True
                        
                        for rank in range(ranks):

                            df_rank = df_params[df_params["rank"]==rank]

                            rank_features = features_data[(rank*n_features):(rank*n_features)+n_features]

                            for r in range(len(rank_features)):

                                name = features[r]
                                val = float(rank_features[r])
                                
                                print(f"{name} -> {val}:{df_rank[name].item()}")

                                if abs(val-df_rank[name].item())>=100:
                                    correct=False
                        if not correct:
                            print(f"Correctness check failed for ({nodes}, {ppn})")
                            return

    print(f"Correctness for {mat_name} passed!")

                

def split(df, size):    
    problems = df['problem'].unique()
    s = int(len(problems)*size)

    test_problems, train_problems = problems[:s],problems[s:]
    
    test, train = df[df['problem'].isin(test_problems)], df[df['problem'].isin(train_problems)]
    return test,train 


def loc_mult_model(X, use_xgb=True):
    X = X.drop(labels=["rank"], axis=1)
    if use_xgb:
        bst = xgb.Booster()
        bst.load_model(f"{path_prefix}models/xgb-mult-best.model")
        return np.array(bst.predict(xgb.DMatrix(X)))
    else:
        return 0


def merge_model(X, use_xgb=True):
    X = X.drop(labels=["rank"], axis=1)
    if use_xgb:
        bst = xgb.Booster()
        bst.load_model(f"{path_prefix}models/xgb-merge-best.model")
        return np.array(bst.predict(xgb.DMatrix(X)))
    else:
        return 0

def bcast_model(X):

    def bcast_tree(grid_dim, msg_size):
        alpha = math.log2(grid_dim)*perlmutter_params.inter_alpha
        beta = math.log2(grid_dim)*(total_bytes / perlmutter_params.inter_beta)
        return (alpha+beta) / 1e6


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
        bcast_time = bcast_tree(grid_dim, total_bytes)
        
        times_mat[row_rank, col_rank] = bcast_time

    row_times = np.sum(times_mat, axis=1)
    col_times = np.sum(times_mat, axis=0)

    for i in range(times_mat.shape[0]):
        for j in range(times_mat.shape[1]):
            times_mat[i,j] = row_times[i] + col_times[j]

    return times_mat.flatten()


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--plotname",type=str)
    parser.add_argument("--label",type=str)
    parser.add_argument("--problem",type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument('--load', const=1, nargs='?', type=int)
    parser.add_argument('--correctness', const=1, nargs='?', type=int)

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
    
    if args.correctness:
        correctness(df, args.problem)
    else:
        eval_spgemm(args, df)
    

    
