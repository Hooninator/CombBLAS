#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=5:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --mem=400GB

# grid 16x16 = 256 mpi process
# mpi process per node 2 
# node 128

export OMP_NUM_THREADS=128

srun /global/homes/y/yuxihong/graphclustering/CombBLAS/build/Applications/spgemm1d $ARCH
srun /global/homes/y/yuxihong/graphclustering/CombBLAS/build/Applications/spgemm1d $EUK
# srun /global/homes/y/yuxihong/graphclustering/CombBLAS/a.out