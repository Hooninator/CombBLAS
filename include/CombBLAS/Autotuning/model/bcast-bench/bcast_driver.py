import subprocess
import os
import argparse
import math


def run(args):

    with open("bcast-out.txt", 'w') as file:
        
        node_range = range(1, args.node_upper+1)
        ppn_range = range(2, 129)
        
        for node in node_range:
            for ppn in ppn_range:
                cmd = f"srun -N {node} --tasks-per-node {ppn} osu_bcast -m {args.msg_upper}"
                print(f"Executing {cmd}...")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                result.check_returncode()
                file.write(result.stdout+"\n")
                file.write(f"----NODES:{node} PPN:{ppn}\n")
                file.write(f"----END OF SAMPLE\n")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--msg_upper", type=int)
    parser.add_argument("--node_upper", type=int)
    
    args = parser.parse_args()
    
    run(args)





