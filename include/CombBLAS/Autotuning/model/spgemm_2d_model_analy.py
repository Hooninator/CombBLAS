
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import time
import argparse
import statistics

from scipy.stats import kendalltau

from data_utils import *


def bcast_model(mat_info: pd.Series) -> float:
    
    ppn = mat_info['PPN']
    nodes = mat_info['Nodes']



def mult_model(mat_info: pd.Series) -> float:
    return 0


def merge_model(mat_info: pd.Series) -> float:
    return 0


def model(mat_info: pd.Series) -> float:
    return bcast_model + mult_model + merge_model
    

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


    print(f"----AVERAGE KT: {kendall_sum/len(problems)}")
    print(f"----AVERAGE RMSE: {rmse_sum/len(problems)}")
    print(f"----AVERAGE DIFF: {diff_sum/len(problems)}s")
    print(f"----NUMBER CORRECT: {n_correct}/{len(problems)}")

    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str)
    parser.add_argument("--plotname", type=str)
    
    args = parser.parse_args()

    df = load_spgemm_2d_data(args.infile)
    df['trial'] = df.apply(lambda row: f"{row['A-name']}x{row['B-name']}", axis=1)
    df['bcast-time'] = df.apply(lambda row: row['bcast-A'] + row['bcast-B'], axis=1)

    eval(args, df, bcast_model, 'bcast-time')
    eval(args, df, mult_model, 'mult-time')
    eval(args, df, merge_model, 'merge-time')
    eval(args, df, model, 'total-time')
    

