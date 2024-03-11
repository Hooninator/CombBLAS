#!/bin/bash
#SBATCH --nodes 1
#SBATCH -A m4646
#SBATCH -C cpu
#SBATCH -t 1:00:00
#SBATCH -q regular
#SBATCH -e error-tuning.out


srun -N 1 --tasks-per-node 64 ~/CombBLAS/build/Applications/autotune matrices/$matname/$matname.mtx ~/CombBLAS/build/Applications/autotuning matrices/$matname/$matname.mtx 0
srun -N 1 --tasks-per-node 64 ~/CombBLAS/build/Applications/autotune matrices/$matname/$matname.mtx ~/CombBLAS/build/Applications/autotuning matrices/$matname/$matname.mtx 1



