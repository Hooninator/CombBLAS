
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import time
import argparse
import statistics
import math

from scipy.stats import kendalltau
from dataclasses import dataclass

from data_utils import *

@dataclass
class PlatformParams:
    inter_beta: float
    inter_alpha: float

perlmutter_params = PlatformParams(2398.54, 3.9)

def bcast_model(mat_info: pd.Series) -> float:
    
    ppn = mat_info.PPN
    nodes = mat_info.Nodes

    total_procs = ppn * nodes

    nnz_estimate_A = mat_info['density-A'] * mat_info['m-A'] * mat_info['n-A']
    nnz_estimate_B = mat_info['density-B'] * mat_info['m-B'] * mat_info['n-B']

    bytes_A = nnz_estimate_A * 8 + nnz_estimate_A * 8 + mat_info['m-A'] * 8
    bytes_B = nnz_estimate_B * 8 + nnz_estimate_B * 8 + mat_info['m-B'] * 8
    
    bcast_time_A = math.log2(total_procs)*perlmutter_params.inter_alpha + math.log2(total_procs)*(bytes_A / perlmutter_params.inter_beta)
    bcast_time_B = math.log2(total_procs)*perlmutter_params.inter_alpha + math.log2(total_procs)*(bytes_B / perlmutter_params.inter_beta)

    return bcast_time_A + bcast_time_B




def mult_model(mat_info: pd.Series) -> float:
    return 0


def merge_model(mat_info: pd.Series) -> float:
    return 0


def model(mat_info: pd.Series) -> float:
    return bcast_model(mat_info) + mult_model(mat_info) + merge_model(mat_info)
    

def eval(args, df, model_fn, phase_name):
    
    problems = list(df['trial'].unique())
    
    kt_sum=diff_sum=rmse_sum=n_correct=0

    for problem in problems:
        df_problem = df[df['trial']==problem]
        y = np.array(df_problem[phase_name])
        y_pred = np.array(df_problem.apply(model_fn, axis=1))

        kt_sum += kendalltau(y, y_pred).correlation
        diff_sum += abs(np.min(y) - y[np.argmin(y_pred)])
        rmse_sum += ((np.linalg.norm(y-y_pred)**2)/len(y))**(1/2)
        n_correct += 1 if np.argmin(y)==np.argmin(y_pred) else 0


    print(f"----AVERAGE KT: {kt_sum/len(problems)}")
    print(f"----TOTAL RMSE: {rmse_sum}")
    print(f"----TOTAL DIFF: {diff_sum}s")
    print(f"----NUMBER CORRECT: {n_correct}/{len(problems)}")

    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str)
    parser.add_argument("--plotname", type=str)
    
    args = parser.parse_args()

    df = load_spgemm2d_data(args.infile)
    df['trial'] = df.apply(lambda row: f"{row['A-name']}x{row['B-name']}", axis=1)
    df['bcast-time'] = df.apply(lambda row: row['bcast-A'] + row['bcast-B'], axis=1)

    eval(args, df, bcast_model, 'bcast-time')
    eval(args, df, mult_model, 'local-mult')
    eval(args, df, merge_model, 'merge')
    eval(args, df, model, 'total-time')
    

