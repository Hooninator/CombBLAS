
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>
#include <cassert>
#include <cstdlib>
#include <sstream>

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

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string(message)); \
        } \
    } while (false)


#define ATIMING

namespace combblas {


namespace autotuning {




enum JobManager {
    M_SLURM
} typedef JobManager;


/* Info about job. Used to establish limits for tunable params */
class JobInfo {
public:
    
    JobInfo(JobManager jobManager) {

        switch(jobManager) {

            case M_SLURM:
            {
                SetJobInfoSlurm();
                break;
            }

            default: 
            {
                throw std::runtime_error("Invalid job manager " + jobManager);
            }
        } 

    }
    
    void SetJobInfoSlurm() {
        
        ASSERT(std::getenv("SLURM_NNODES")!=nullptr, "Are you sure you're using slurm?");
        ASSERT(std::getenv("SLURM_NTASKS_PER_NODE")!=nullptr, "Please use the --tasks-per-node option");

        nodes = std::atoi(std::getenv("SLURM_NNODES")); 
        tasksPerNode = std::atoi(std::getenv("SLURM_NTASKS_PER_NODE")); 
        totalTasks = std::atoi(std::getenv("SLURM_NTASKS")); 
        
        if (std::getenv("SLURM_GPUS")!=nullptr) 
            totalGPUs = std::atoi(std::getenv("SLURM_GPUS")); 
        if (std::getenv("SLURM_GPUS_PER_NODE")!=nullptr) 
            gpusPerNode = std::atoi(std::getenv("SLURM_GPUS_PER_NODE")); 
    }

    int nodes;

    int tasksPerNode;
    int totalTasks;

    int gpusPerNode;
    int totalGPUs; 
    
};    


template <typename T>
class SearchSpace {


};




class AutotunerSpGEMM3D {

public:
    
    /* Output of tune routine. Struct of relevant SpGEMM3D params that should be used to create a CommGrid3D */
    class SpGEMM3DParams {
    public:
        
        SpGEMM3DParams(int nodes, int ppn, int layers):
        nodes(nodes), ppn(ppn), layers(layers) {}
        
        void print() {
            std::cout<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
        }
        
        std::string outStr() {
            std::stringstream ss;
            ss<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
            return ss.str();
        }

        int nodes;
        int ppn;
        int layers; 
    }; 
    
    /* CONSTRUCTORS */
    
    //Calls measuring routines to create PlatformParams instance
    AutotunerSpGEMM3D(JobManager jobManager):
    platformParams(PlatformParams()), jobInfo(JobInfo(jobManager)) 
    {
        
        SetMPIInfo();

    }
    
    // Assumes PlatformParams has already been constructed
    AutotunerSpGEMM3D(PlatformParams& params, JobManager jobManager):
    platformParams(params), jobInfo(JobInfo(jobManager))
    {
        SetMPIInfo(); 
    }
    
    
    void SetMPIInfo() {

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);

    }
    
    
    
    //TODO: Need member functions that estimate nnz per proc in 3D grid without actually creating the 3D grid
    //actually creating the grid is likely slow if done lots of times
    //will handle this in SymbolicSpParMat3D

    
    /* TUNING */

    /* Main tuning routine for CPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::shared_ptr<CommGrid3D> Tune(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B){
        
        auto grid = A.getcommgrid(); 
        
        SpGEMM3DParams resultParams = SearchNaive(A,B,grid); 
        
        auto resultGrid = MakeGridFromParams(resultParams);
        
        return resultGrid;

    }

    /* Main tuning routine for GPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::shared_ptr<CommGrid3D> TuneGPU() {/*TODO*/}

    
    /* SEARCH SPACE EXPLORATION */
    
    /* Given a set of parameters, construct a 3D processor grid from a communicator that only contains the processes
     * with local ranks < ppn on nodes < n
     */
    std::shared_ptr<CommGrid3D> MakeGridFromParams(SpGEMM3DParams params) {
        int nodeRank = (rank / jobInfo.tasksPerNode);
        int color = static_cast<int>(nodeRank < params.nodes && localRank < params.ppn);
        int key = rank;
        
        MPI_Comm newComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);
        
        int newSize;
        MPI_Comm_size(newComm, &newSize);

        if (color==1) {

            ASSERT(newSize==params.nodes*params.ppn, "Expected communicator size to be " + std::to_string(params.nodes*params.ppn) +
            ", but it was " + std::to_string(newSize));
            ASSERT(IsPerfectSquare(newSize / params.layers), "Each 2D grid must be a perfect square, instead got " + params.outStr());
            
            
            std::shared_ptr<CommGrid3D> newGrid;
            newGrid.reset(new CommGrid3D(newComm, params.layers, 0, 0));
            
            return newGrid;

        } else {
            return NULL;
        }

    }


    /* Return all powers of 2 node counts up to the number of nodes allocated to the job */
    std::vector<int> GetValidNNodes() {
        int i=0;
        std::vector<int> nodesVec(static_cast<int>(log2(jobInfo.nodes))+1);
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
    
    /* Brute force search. Explicitly creates 3D comm grid for each paramter combo */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams SearchNaive(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, std::shared_ptr<CommGrid> grid) {
        
#ifdef ATIMING
    auto stime1 = MPI_Wtime();
#endif
    
    size_t searchSize = 0;
    SpGEMM3DParams minParams {jobInfo.nodes, jobInfo.tasksPerNode, 1};
    double minTime = EstimateRuntime(A, B, minParams);

    auto nodesVec = GetValidNNodes();

    for (auto const& nodes : nodesVec) {

        auto ppnVec = GetValidPPNs(nodes);

        for (auto const& ppn : ppnVec) {

            auto layerVec = GetValidNLayers(ppn, nodes);

            for (auto const& layer : layerVec) {

                SpGEMM3DParams currParams {nodes, ppn, layer};
                double currTime = EstimateRuntime(A,B,currParams);
                if (currTime<minTime) minParams = currParams;                
                searchSize+=1;

            }
        }
    } 

#ifdef ATIMING
    auto etime1 = MPI_Wtime();
    if (rank==0) std::cout<<"[SearchNaive] Total time: "<<(etime1-stime1)<<"s"<<std::endl;
    if (rank==0) std::cout<<"[SearchNaive] Search size: "<<searchSize<<std::endl;
#endif
    
    return minParams;
    
    }
    
    
    /* Brute force search in parallel. Explicitly creates 3D comm grid for each paramter combo */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams ParallelSearchNaive(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, std::shared_ptr<CommGrid> grid) {
        
#ifdef ATIMING
    auto stime1 = MPI_Wtime();
#endif
    
    

#ifdef ATIMING
    auto etime1 = MPI_Wtime();
    if (rank==0) std::cout<<"[ParallelSearchNaive] Total time: "<<(etime1-stime1)<<"s"<<std::endl;
#endif
    
    }
    
    
    /* RUNTIME ESTIMATION */
    
    /* Get runtime estimate of a certain combo of parameters */    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, SpGEMM3DParams params) {
        /* TODO */
        MakeGridFromParams(params);
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
    
    
    int GetRank() const {return rank;}
    int GetLocalRank() const {return localRank;}
    int GetWorldSize() const {return worldSize;}
    
    /* UTILITY FUNCTIONS */

    bool IsPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root*root==num;
    }
    

    ~AutotunerSpGEMM3D(){}

private:
    PlatformParams platformParams;
    JobInfo jobInfo;
    int rank; int worldSize;
    int localRank;

};//AutotunerSpGEMM3D


}//autotuning
}//combblas
