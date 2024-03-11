#!/usr/bin/bash

#SBATCH --nodes 1
#SBATCH -A m4646
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -C cpu
#SBATCH -e error1gnn.out

stime=$SECONDS
tlimit=$((175*60))

for FILE in $(ls $PSCRATCH/matrices | shuf); do 

    ctime=$SECONDS

    # This is necessary to prevent a bunch of errors when the job time expires
    if (( ctime - stime >= tlimit)); then
        echo "$((tlimit / 60)) minutes have passed, stopping trials pre-emptively"
        break
    fi

    matname=$(echo $FILE | grep -o '[^/]*$')
    echo "$matname"

    if [ "$1" == "run-combblas" ]; then

        if [ "$2" == "xgb" ]; then
            log=log-xgb.out
            prefix=samples-xgb
        fi
        if [ "$2" == "gnn" ]; then
            log=log-gnn.out
            prefix=samples-gnn
        fi

        if grep -q "${matname}-${SLURM_NNODES}" $log; then
            echo "${matname}-${SLURM_NNODES} already executed"
            continue
        fi

        module load python/3.9

        python3 driver.py --alg 2D --matA $matname --matB $matname --code 3 --nodes $SLURM_NNODES --permute 0 --model $2
        code1=$?

        python3 driver.py --alg 2D --matA $matname --matB $matname --code 3 --nodes $SLURM_NNODES --permute 1 --model $2
        code2=$?

        # Avoid writing incomplete samples
        #if [ "$code1" -eq 0 ] && [ "$code2" -eq 0]; then
        cat $prefix-${SLURM_NNODES}${matname}${matname}.txt>>$prefix-$SLURM_NNODES.txt
        #fi

        echo "${matname}-${SLURM_NNODES}">>$log
        rm -f $prefix-${SLURM_NNODES}${matname}${matname}.txt

    fi

    if [ "$1" == "run-autotuning" ]; then
        cd ~/CombBLAS/build 
        srun -N 1 --tasks-per-node 64 Applications/autotune $PSCRATCH/matrices/$matname/$matname.mtx $PSCRATCH/matrices/$matname/$matname.mtx 0
        cd ~/CombBLAS/include/CombBLAS/Autotuning/model
    fi
done

