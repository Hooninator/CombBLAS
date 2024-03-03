import argparse
import subprocess
import os



def write_output(args, result_lst):
    
    fname = f"combblas-{args.alg}-{args.code}-N{args.nodes}-{args.matA}x{args.matB}.out"

    file = open(fname, "w")

    for result in result_lst:
        file.write(result[0]+"\n") #cmd
        file.write(result[1]) #output
     
    file.close()
    
    os.rename(f"./{fname}", f"./perlmutter-dat/{fname}")


def get_layers(ppn, nodes):
    n = ppn*nodes
    layers = [1]
    for l in [2, 4, 8, 16, 32, 64, 128, 256]:
        if l>n:
            continue
        grid_size_2d = n // l
        if (round(grid_size_2d**(1/2))**2==grid_size_2d): # perfect square 2d grids
            layers.append(l)
    print("Layers: " + str(layers))
    return layers


def run(args):
    
    combblas_cmd = f" combblas-spgemm {args.alg} $PSCRATCH/matrices/{args.matA}/{args.matA}.mtx $PSCRATCH/matrices/{args.matB}/{args.matB}.mtx args.code "
    
    cmd_lst = []

    for t in [2**i for i in range(0,6)]:
        ppn = 128//t
        n = args.nodes * ppn
        if (round(n**(1/2))**2!=n):
            continue
        c = (128//ppn)*2
        srun_cmd = f"export OMP_NUM_THREADS={t} && srun --tasks-per-node {ppn} -N {args.nodes}"
        if args.alg=="3D":
            layers = get_layers(ppn, args.nodes)
            for l in layers:
                combblas_cmd_tmp = combblas_cmd + str(l)
                cmd  = srun_cmd + combblas_cmd_tmp 
                cmd_lst.append(cmd)
        else:
            cmd = srun_cmd + combblas_cmd
            cmd_lst.append(cmd)
    
    print(cmd_lst)
    
    result_lst = []

    for cmd in cmd_lst:
        print(f"Executing {cmd}")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        result.check_returncode()
        result_lst.append((cmd,result.stdout))
    
    return result_lst


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str)
    parser.add_argument("--matA", type=str)
    parser.add_argument("--matB", type=str)
    parser.add_argument("--code", type=int)
    parser.add_argument("--nodes", type=int)
    args = parser.parse_args()
    result_lst = run(args)
    #write_output(args, result_lst)
