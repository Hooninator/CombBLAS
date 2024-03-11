#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH -C gpu
#SBATCH -A m4646_g
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -o gnn5.out

module load pytorch
python3 spgemm_2d_model_gnn.py --plotname gnn-$1 --label total-time --train --eval --epochs 1000 --hparams hyperparams/hp$1.txt --modelname gnn-$1
