#!/bin/bash
#SBATCH --nodes 4
#SBATCH -A m4646
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -C cpu
#SBATCH -e error.out


module load python/3.9
python3 driver.py --alg 2D --matA $1 --matB $2 --code 3 --nodes 4 --permute 0
python3 driver.py --alg 2D --matA $1 --matB $2 --code 3 --nodes 4 --permute 1
