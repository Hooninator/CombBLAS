


#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>
#include <cassert>
#include <cstdlib>

#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "CommGrid3D.h"

#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"
#include "mtSpGEMM.h"
#include "MultiwayMerge.h"
#include "CombBLAS.h"

#include "SpParMat3D.h"
#include "SpParMat.h"
#include "PlatformParams.h"
#include "SymbolicSpParMat3D.h"



namespace combblas {


namespace autotuning {


/* Info about job. Used to establish limits for tunable params */
struct JobInfo {

    int nodes;

    int tasksPerNode;
    int totalTasks;

    int gpusPerNode;
    int totalGPUs; 
    
    void SetJobInfoSlurm() {
        assert(std::getenv("SLURM_NNODES")!=nullptr);
        nodes = std::atoi(std::getenv("SLURM_NNODES")); 
        tasksPerNode = std::atoi(std::getenv("SLURM_NTASKS_PER_NODE")); 
        totalTasks = std::atoi(std::getenv("SLURM_NTASKS")); 
        gpusPerNode = std::atoi(std::getenv("SLURM_GPUS_PER_NODE")); 
        totalGPUs = std::atoi(std::getenv("SLURM_GPUS")); 
    }
    
} typedef JobInfo;    


enum JobManager {
    M_SLURM
} typedef JobManager;


class AutotunerSpGEMM3D {

public:
    
    /* Output of tune routine. Struct of relevant SpGEMM3D params that should be used to create a CommGrid3D */
    struct SpGEMM3DParams {
        int nodes;
        int ppn;
        int layers; 
    } typedef SpGEMM3DParams;
    
    
    /* CONSTRUCTORS */
    
    //Calls measuring routines to create PlatformParams instance
    AutotunerSpGEMM3D(JobManager jobManager):
    platformParams(PlatformParams()), jobInfo(JobInfo()) 
    {
        
        /* Set job params based on workload manager */
        switch(jobManager) {

            case M_SLURM:
            {
                jobInfo.SetJobInfoSlurm();
                break;
            }

            default: 
            {
                throw std::runtime_error("Invalid job manager " + jobManager);
            }
        } 

    }
    
    // Assumes PlatformParams has already been constructed
    AutotunerSpGEMM3D(PlatformParams& params, JobManager jobManager):
    platformParams(params), jobInfo(JobInfo())
    {
    
        /* Set job params based on workload manager */
        switch(jobManager) {

            case M_SLURM:
            {
                jobInfo.SetJobInfoSlurm();
                break;
            }

            default: 
            {
                throw std::runtime_error("Invalid job manager " + jobManager);
            }
        } 

    }
    
    
    /* SEARCH SPACE EXPLORATION */
    

    /* Return all powers of 2 node counts up to the number of nodes allocated to the job */
    std::vector<int> GetValidNNodes() {
        int i=0;
        std::vector<int> nodesVec;
        nodesVec.reserve(static_cast<int>(log2(jobInfo.nodes)));
        std::generate(nodesVec.begin(), nodesVec.end(), [&i]() mutable {return pow(2, i++);});
        return nodesVec;
    }

    
    /* Given node count, return valid PPN values. Valid if perfect square number of tasks */
    std::vector<int> GetValidPPNs(int nodes) {
        std::vector<int> ppnVec;
        ppnVec.reserve(static_cast<int>(log2(jobInfo.tasksPerNode)));
        for(int ppn=1; ppn<=jobInfo.tasksPerNode; ppn*=2) {
            if (IsPerfectSquare(ppn*nodes)) {
                ppnVec.push_back(ppn);
            }
        }
        return ppnVec;
    }

    
    /* Given node count and ppn value, return all valid values for the layers param. 
     * A value for the layers param is valid if the 2D grids are perfect squares.
     */
    std::vector<int> GetValidNLayers(int ppn, int nodes) {
        int ntasks = ppn*nodes;
        std::vector<int> layerVec;
        layerVec.reserve(static_cast<int>(log2(ntasks)));
        for (int l=2; l<=ntasks; l*=2) {
            int gridSize = ntasks / l;
            if (IsPerfectSquare(gridSize)) {
                layerVec.push_back(l);
            }
        }
        return layerVec;
    }
    
    
    //TODO: Need member functions that estimate nnz per proc in 3D grid without actually creating the 3D grid
    //actually creating the grid is likely slow if done lots of times
    //will handle this in SymbolicSpParMat3D

    
    /* TUNING */

    /* Main tuning routine for CPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams Tune(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B){
        
        auto grid = A.getcommgrid(); 
        
        SpGEMM3DParams tunableParams; 
        
        /* TODO: Efficient parameter space searching method */

    }

    /* Main tuning routine for GPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams TuneGPU() {/*TODO*/}


    
    /* RUNTIME ESTIMATION */
    
    /* Get runtime estimate of a certain combo of parameters */    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, SpGEMM3DParams params) {
        /* TODO */
        return 0;
    }

    /* Time estimates for each step of 3D SpGEMM */    
    /* TODO: All of these */
    double ABcastTime(){return 0;}
    double BBcastTime(){return 0;}
    double LocalMultTime(){return 0;}
    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;} 
    
    
    
    /* UTILITY FUNCTIONS */

    bool IsPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root*root==num;
    }

    ~AutotunerSpGEMM3D(){}

private:
    PlatformParams platformParams;
    JobInfo jobInfo;

};//AutotunerSpGEMM3D


}//autotuning
}//combblas
