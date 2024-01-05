
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
#include "Logger.h"

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string(message)); \
        } \
    } while (false)


//#define PROFILE
//#define DEBUG

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

JobInfo *jobPtr = nullptr;
Logger *debugPtr = nullptr;
Logger *statPtr = nullptr;

void Init(JobManager jm) {
    
    int initialized;
    MPI_Initialized(&initialized);
    ASSERT(initialized==1, "Please call MPI_Init() before calling this method");
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    
    jobPtr = new JobInfo(jm);

#ifdef DEBUG
    debugPtr = new Logger(rank,"logfile"+std::to_string(rank)+".out");
#endif

#ifdef PROFILE
    statPtr = new Logger(rank, "statfile-N"+std::to_string(jobPtr->nodes)+".out");
#endif    

    initCalled = true;
}



/* Generic model for communication time */
class CommModel {

public:
    CommModel(){}
    
    virtual double ComputeTime() {throw std::runtime_error("This method should never be called");}
    
    virtual int GetWorld() {throw std::runtime_error("This method should never be called");}

};

/* T = alpha + bytes/beta */
template <typename IT>
class PostCommModel : public CommModel {

public:
    
    PostCommModel(double alpha, double beta, std::function<int()> ComputeNumMsgs, std::function<IT()> ComputeNumBytes):
    alpha(alpha), beta(beta), ComputeNumMsgs(ComputeNumMsgs), ComputeNumBytes(ComputeNumBytes)
    {
        
    }
    
    inline double ComputeTime() {
        return ComputeNumMsgs() * alpha + (ComputeNumBytes())/beta;
    }

private:
    double alpha; double beta;
    std::function<int()> ComputeNumMsgs; std::function<IT()> ComputeNumBytes;

};



template <typename IT, typename NT, typename DER>
class SpGEMM3DMatrixInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;
    
    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& M):
    nnz(M.getnnz()), ncols(M.getncol()), nrows(M.getnrow()) {
        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
    }
    
    SpGEMM3DMatrixInfo(IT nnz, IT cols, IT rows):
    nnz(nnz), ncols(cols), nrows(rows) {
        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
    }
    
    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}
    
    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}
    
    inline float GetDensity() const {return density;}
    
private:

    
    IT nnz;
    IT ncols;
    IT nrows;
    
    float density;
        
};


template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM3DInputs {

public:

    SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo, 
                                                SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo):
    Ainfo(Ainfo), Binfo(Binfo) {}

    SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo;
    SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo;
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
        ss<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")";
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
    double EstimateRuntime(SpGEMM3DInputs<AIT,ANT,ADER, BIT, BNT, BDER> inputs, PlatformParams& platformParams) {

#ifdef DEBUG
        debugPtr->Log(OutStr());
#endif
        
        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo; 
        
        CommModel *bcastModelA = MakeBcastModelPost(Ainfo, platformParams, BCAST_TREE);
        CommModel *bcastModelB = MakeBcastModelPost(Binfo, platformParams, BCAST_TREE);
        
        double bcastTime = BcastTime(bcastModelA) + BcastTime(bcastModelB); 
        
#ifdef PROFILE
        statPtr->Print("[Bcast time] " + std::to_string(bcastTime/1e6) + "s");
        statPtr->Log("Bcast time " + std::to_string(bcastTime/1e6) + "s");
#endif
        delete bcastModelA; delete bcastModelB;

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
    PostCommModel<IT> * MakeBcastModelPost(SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, PlatformParams &params, BcastAlgorithm alg) {
        
        std::function<int()> _ComputeNumMsgs;
        std::function<IT()> _ComputeNumBytes;

        switch(alg) {

            case BCAST_TREE:
            {
                _ComputeNumMsgs = [this]() {
                    int bcastWorldSize = static_cast<int>(sqrt( (this->nodes*this->ppn) / this->layers ));
                    return static_cast<int>(log2(bcastWorldSize));
                }; 
                
                _ComputeNumBytes = [Minfo, this]() {
                    int totalProcs = this->nodes * this->ppn;

                    //TODO: Make these things methods in the SpGEMM3DMatrixInfo class
                    IT localNcols = Minfo.GetNcols() / static_cast<int>(sqrt(totalProcs));
                    IT localNrows = Minfo.GetNrows() / static_cast<int>(sqrt(totalProcs));
                    IT localMatSize = localNcols * localNrows;

                    IT localNnzApprox = static_cast<int>(Minfo.GetDensity() * localMatSize); 
                    IT sendBytes = localNnzApprox * Minfo.GetNzvalSize() +
                                    localNnzApprox * Minfo.GetIndexSize() +
                                    (localNnzApprox + 1) * Minfo.GetIndexSize();

                    int bcastWorldSize = static_cast<int>(sqrt(this->nodes*this->ppn / this->layers));
                    IT totalBytes = sendBytes*static_cast<int>(log2(bcastWorldSize));
#ifdef DEBUG
                    debugPtr->Log("bcastWorldSize: " + std::to_string(bcastWorldSize));
                    debugPtr->Log("Local mat size: " + std::to_string(localMatSize));
                    debugPtr->Log("Local nnz estimate: " + std::to_string(localNnzApprox));
                    debugPtr->Log("Send bytes estimate: " + std::to_string(totalBytes));
#endif                    
                    return totalBytes;
                };
                
                break;
            }

        }
        
        return new PostCommModel<IT>(params.GetInternodeAlpha(), params.GetInternodeBeta(), 
                                    _ComputeNumMsgs, _ComputeNumBytes);

    }


    double BcastTime(CommModel * bcastModel) {

#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif

        double singleBcastTime = bcastModel->ComputeTime();

        int gridSize = (nodes*ppn) / layers; 
        double finalTime = singleBcastTime * sqrt(gridSize); 

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Print("[Bcast calc time] " + std::to_string(t1) + "s");
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
#endif
        return finalTime;
    }



    double LocalMultTime(){return 0;}
    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;} 


    /* Given a set of parameters, construct a 3D processor grid from a communicator that only contains the processes
     * with local ranks < ppn on nodes < n
     */
    std::shared_ptr<CommGrid3D> MakeGridFromParams() {
        //TODO: This needs some major work
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
    
        SpGEMM3DMatrixInfo<AIT,ANT,ADER> Ainfo(A3D);
        SpGEMM3DMatrixInfo<BIT,BNT,BDER> Binfo(B3D);
        
        SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(Ainfo, Binfo);
        
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

#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif

        auto searchSpace = P::ConstructSearchSpace();
        ASSERT(searchSpace.size()>0, "Search space is of size 0!");

#ifdef PROFILE
        statPtr->Log("Search space size: " + std::to_string(searchSpace.size()));
#endif

        P bestParams;  
        double bestTime = std::numeric_limits<double>::max(); 

        for (P currParams : searchSpace) {
#ifdef PROFILE
            statPtr->Log(currParams.OutStr());
#endif
            double currTime = currParams.EstimateRuntime(input, platformParams);
            if (currTime<=bestTime) {
                bestTime = currTime;
                bestParams = currParams;
            }
#ifdef PROFILE
            statPtr->Log("Total runtime " + std::to_string(currTime)+"s");
#endif
        }

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Print("[SearchBruteForce] " + std::to_string(t1) + "s");
        statPtr->Print("SearchBruteForce time " + std::to_string(t1) + "s");
#endif
        
        return bestParams;
    }
    

    ~Autotuner(){}

private:
    PlatformParams platformParams;

};//Autotuner


}//autotuning
}//combblas
