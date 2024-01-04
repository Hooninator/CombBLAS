
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <limits>

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
#include "Debugger.h"

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string(message)); \
        } \
    } while (false)


#define ATIMING
#define DEBUG

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


//Global variables

int rank; int worldSize;
int localRank;
bool initCalled = false;

Debugger *debugPtr = nullptr;
JobInfo *jobPtr = nullptr;

void Init(JobManager jm) {
    
    int initialized;
    MPI_Initialized(&initialized);
    ASSERT(initialized==1, "Please call MPI_Init() before calling this method");
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    
    debugPtr = new Debugger(rank);
    jobPtr = new JobInfo(jm);
    
    initCalled = true;
}



/* Generic model for communication time */
class CommModel {

public:
    CommModel(){}
    
    virtual double ComputeTime() {throw std::runtime_error("This method should never be called");}
    
    virtual MPI_Comm GetWorld() {throw std::runtime_error("This method should never be called");}

};

/* T = alpha + bytes/beta */
class PostCommModel : public CommModel {

public:
    
    PostCommModel(double alpha, double beta, std::function<int()> ComputeNumMsgs, std::function<int()> ComputeNumBytes,
                    MPI_Comm world):
    alpha(alpha), beta(beta), ComputeNumMsgs(ComputeNumMsgs), ComputeNumBytes(ComputeNumBytes), world(world)
    {
        
    }
    
    inline double ComputeTime() {
        return ComputeNumMsgs() * alpha + (ComputeNumBytes())/beta;
    }

    inline MPI_Comm GetWorld() {return world;}

private:
    double alpha; double beta;
    std::function<int()> ComputeNumMsgs; std::function<int()> ComputeNumBytes;

    MPI_Comm world;

};



template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM3DInput {
public:
    SpParMat3D<AIT, ANT, ADER>& A;
    SpParMat3D<BIT, BNT, BDER>& B;
};

/* Output of tune routine. Relevant SpGEMM3D params that should be used to create a CommGrid3D */
class SpGEMM3DParams {
public:
    
    SpGEMM3DParams(){}    


    SpGEMM3DParams(int nodes, int ppn, int layers):
    nodes(nodes), ppn(ppn), layers(layers) {}

    
    void Print() {
        std::cout<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
    }

    
    std::string OutStr() {
        std::stringstream ss;
        ss<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
        return ss.str();
    }


    static std::vector<SpGEMM3DParams> ConstructSearchSpace() {
        std::vector<SpGEMM3DParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=jobPtr->tasksPerNode; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes)) {
                    for (int _layers=1; _layers<=_ppn*_nodes; _layers*=2) {
                        int gridSize = (_ppn*_nodes) / _layers;
                        if (IsPerfectSquare(gridSize))
                            result.push_back(SpGEMM3DParams(_nodes,_ppn,_layers));
                    }
                }
            }
        }
        return result;
    }


    /* Get runtime estimate of a certain combo of parameters */    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpGEMM3DInput<AIT,ANT,ADER,BIT,BNT,BDER> input, PlatformParams platformParams) {
        /* TODO: Make new commgrid for A,B based on params */

#ifdef DEBUG
        debugPtr->Log(OutStr());
#endif
        
        auto paramGrid = MakeGridFromParams();
        
        if (paramGrid==nullptr) {
            debugPtr->Log("Not participating in this one");
            return -1.0;
        }

#ifdef DEBUG
        debugPtr->Log("Made grid");
#endif

        //SpParMat3D<AIT,ANT,ADER> A(input.A.seqptr(), paramGrid, input.A.isColSplit());

#ifdef DEBUG
        debugPtr->Log("Made A");
#endif

        //SpParMat3D<BIT,BNT,BDER> B(input.B.seqptr(), paramGrid, input.B.isColSplit());

#ifdef DEBUG
        debugPtr->Log("Made B");
#endif

        
        auto grid = paramGrid; 
        
        //CommModel *bcastModelA = MakeBcastModelPost(A, platformParams, BCAST_TREE,  
         //                           grid->GetCommGridLayer()->GetRowWorld());
        //CommModel *bcastModelB = MakeBcastModelPost(B, platformParams, BCAST_TREE, 
           //                         grid->GetCommGridLayer()->GetColWorld());
        
#ifdef DEBUG
        debugPtr->Log("Made models");
        debugPtr->Log(grid);
#endif

        double bcastTime = 0.0;//BcastTime(bcastModelA, grid) + BcastTime(bcastModelB, grid); 
        
#ifdef ATIMING
        debugPtr->Print("[Bcast time] " + std::to_string(bcastTime) + "s");
#endif

        //MPI_Barrier(grid->GetWorld());        

        //delete bcastModelA; delete bcastModelB;

#ifdef DEBUG
        debugPtr->Log("Done with runtime estimation");
#endif
        
        return 0;
    }


    /* BROADCAST MODELS */    
    
    enum BcastAlgorithm {
        BCAST_TREE,
        BCAST_RING
    } typedef BcastAlgorithm;
    
    enum BcastWorld {
        BCAST_ROW,
        BCAST_COL
    } typedef BcastWorld;
    
    
    template <typename IT, typename NT, typename DER>
    PostCommModel * MakeBcastModelPost(SpParMat3D<IT,NT,DER>& M, PlatformParams &params, BcastAlgorithm alg, MPI_Comm bcastWorld) {
        
        std::function<int()> _ComputeNumMsgs;
        std::function<int()> _ComputeNumBytes;

        auto layerMat = M.GetLayerMat();

        switch(alg) {

            case BCAST_TREE:
            {
                _ComputeNumMsgs = [bcastWorld]() {
                    int bcastWorldSize; 
                    MPI_Comm_size(bcastWorld, &bcastWorldSize);
                    return static_cast<int>(log2(bcastWorldSize));
                }; 
                
                _ComputeNumBytes = [&layerMat]() {
                    // Size of message this processor sends in bcasts        
                    int sendBytes = layerMat->seqptr()->getnnz() * sizeof(NT) + //nnz size
                                      layerMat->seqptr()->getncol() * sizeof(IT) + //colptr size
                                      (layerMat->seqptr()->getnrow() + 1 ) * sizeof(IT); //rowindices size
                    return sendBytes;
                };
                
                break;
            }

        }
        
        return new PostCommModel(params.GetInternodeAlpha(), params.GetInternodeBeta(), 
                                    _ComputeNumMsgs, _ComputeNumBytes,
                                    bcastWorld);

    }


    double BcastTime(CommModel * bcastModel, std::shared_ptr<CommGrid3D> grid) {

#ifdef ATIMING
        auto stime1 = MPI_Wtime();
#endif

        double localBcastTime = bcastModel->ComputeTime();

#ifdef DEBUG
        debugPtr->Log("Computed local Bcast time");
#endif

        double * totalTime = new double(0.0);
        MPI_Allreduce(&localBcastTime, totalTime, 1, MPI_DOUBLE, MPI_SUM, bcastModel->GetWorld()); 
        
        double * finalTime = new double(0.0);
        MPI_Allreduce(totalTime, finalTime, 1, MPI_DOUBLE, MPI_MAX, grid->GetWorld());

#ifdef ATIMING
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        debugPtr->Print("[Bcast calc time] " + std::to_string(t1) + "s");
#endif
        return *finalTime;
    }



    double LocalMultTime(){return 0;}
    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;} 


    /* Given a set of parameters, construct a 3D processor grid from a communicator that only contains the processes
     * with local ranks < ppn on nodes < n
     */
    std::shared_ptr<CommGrid3D> MakeGridFromParams() {
        int nodeRank = (autotuning::rank / jobPtr->tasksPerNode);
        int color = static_cast<int>(nodeRank < nodes && autotuning::localRank < ppn);
        int key = autotuning::rank;
        
        MPI_Comm newComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);
        
        int newSize;
        MPI_Comm_size(newComm, &newSize);

        if (color==1) {

            ASSERT(newSize==nodes*ppn, 
            "Expected communicator size to be " + std::to_string(nodes*ppn) + ", but it was " 
            + std::to_string(newSize));

            ASSERT(IsPerfectSquare(newSize / layers), 
            "Each 2D grid must be a perfect square, instead got " + OutStr());
            
            std::shared_ptr<CommGrid3D> newGrid;
            newGrid.reset(new CommGrid3D(newComm, layers, 0, 0));

#ifdef DEBUG
            debugPtr->Log("Grid size "+std::to_string(newSize));
#endif
            
            return newGrid;

        } else {
            return NULL;
        }

    }


    /* UTILITY FUNCTIONS */

    static bool IsPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root*root==num;
    }


    /* Tunable parameters */

    int nodes;
    int ppn;
    int layers; 


};//SpGEMM3DParams 



enum TuningMethod {
    BRUTE_FORCE
}typedef TuningMethod;

class Autotuner {

public:
    
    
    /* CONSTRUCTORS */
    
    //Calls measuring routines to create PlatformParams instance
    Autotuner(): platformParams(PlatformParams()) {
        ASSERT(initCalled, "Please call autotuning::Init() first.");
    }
    

    // Assumes PlatformParams has already been constructed
    Autotuner(PlatformParams& params): platformParams(params) {
        ASSERT(initCalled, "Please call autotuning::Init() first.");
    }
    
    
    //TODO: Need member functions that estimate nnz per proc in 3D grid without actually creating the 3D grid
    //actually creating the grid is likely slow if done lots of times
    //will handle this in SymbolicSpParMat3D

    
    /* TUNING */
    

    /* Main tuning routine for CPU 3DSpGEMM */
    //TODO: Make the tuning method parameter a std::function instance
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams TuneSpGEMM3D(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, TuningMethod method){
        
        SpParMat3D<AIT, ANT, ADER> A3D(A, 1, true, false);
        SpParMat3D<BIT, BNT, BDER> B3D(B, 1, false, false);
        
        SpGEMM3DInput<AIT,ANT,ADER,BIT,BNT,BDER> inputs{A3D,B3D};
        
        SpGEMM3DParams resultParams; 
        
        switch(method) {
            case BRUTE_FORCE:
            {
                resultParams = SearchBruteForce<SpGEMM3DParams>(inputs); 
                break;
            }
            default:
            {
                break;
            }
        }        

        return resultParams;

    }


    /* Main tuning routine for GPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams TuneSpGEMM3DGPU() {/*TODO*/}

    
    template <typename P, typename I>
    P SearchBruteForce(I input) {

#ifdef ATIMING
        auto stime1 = MPI_Wtime();
#endif

        auto searchSpace = P::ConstructSearchSpace();
        ASSERT(searchSpace.size()>0, "Search space is of size 0!");

        P bestParams;  
        double bestTime = std::numeric_limits<double>::max(); 

        for (P currParams : searchSpace) {
            double currTime = currParams.EstimateRuntime(input, platformParams);
#ifdef DEBUG
            debugPtr->Log("Finished estimation");
#endif
            if (currTime<=bestTime) {
                bestTime = currTime;
                bestParams = currParams;
            }

#ifdef DEBUG
            debugPtr->Log("Finished iteration");
#endif

        }

#ifdef ATIMING
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        debugPtr->Print("[SearchBruteForce] " + std::to_string(t1) + "s");
#endif
        
        return bestParams;
    }
    

    ~Autotuner(){}

private:
    PlatformParams platformParams;

};//Autotuner


}//autotuning
}//combblas
