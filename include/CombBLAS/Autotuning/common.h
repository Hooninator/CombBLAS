#ifndef COMMON_H
#define COMMON_H

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
#include <functional>
#include <exception>
#include <string>
#include <algorithm>
#include <numeric>
#include <utility>
#include <tuple>
//#include <upcxx/upcxx.hpp>

#include "CombBLAS/SpMat.h"
#include "CombBLAS/SpTuples.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/CommGrid.h"                                                                                     
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/MPIType.h"
#include "CombBLAS/LocArr.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/Deleter.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/Friends.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/DistEdgeList.h"
#include "CombBLAS/mtSpGEMM.h"
#include "CombBLAS/MultiwayMerge.h"
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/SpParMat.h"
#include "Logger.h"


#define PROFILE 
//#define DEBUG

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string(message)); \
        } \
    } while (false)


#define INVALID_CALL_ERR() throw std::runtime_error("This method should never be called");

#define UNREACH_ERR() throw std::runtime_error("Never should have come here...");

#ifdef PROFILE

#define INIT_TIMER() auto stime = MPI_Wtime(); auto etime = MPI_Wtime();
#define START_TIMER() stime = MPI_Wtime();
#define END_TIMER(message) \
    do { \
        etime = MPI_Wtime(); \
        statPtr->Print(message + std::to_string(etime - stime)); \
        statPtr->Log(message + std::to_string(etime - stime)); \
    } while (false)

#else

#define INIT_TIMER() 
#define START_TIMER() 
#define END_TIMER(message) 

#endif

#ifdef DEBUG

#define DEBUG_PRINT(message) if(rank==0) debugPtr->Print(message);
#define DEBUG_LOG(message) debugPtr->Log(message);

#else

#define DEBUG_PRINT(message) 
#define DEBUG_LOG(message) 

#endif


namespace combblas {
namespace autotuning {

enum jobManager {
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
        totalTasks = tasksPerNode*nodes;

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
    ASSERT(initialized==1, "Please call MPI_Init() before calling this function");

    //upcxx::init();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);

    jobPtr = new JobInfo(jm);

#ifdef DEBUG
    debugPtr = new Logger(rank,"logfile"+std::to_string(rank)+".out", true);
    debugPtr->Print0("Debug mode active");
#endif


    initCalled = true;
}



void Finalize() {
    
    ASSERT(initCalled, "Please call autotuning::Init() first");

#ifdef PROFILE
    delete statPtr;
#endif

#ifdef DEBUG
    delete debugPtr;
#endif

    delete jobPtr;
    
    //upcxx::finalize();

}


/* UTILITY FUNCTIONS */
std::string ExtractMatName(const std::string& path) {
    size_t start = path.rfind('/') + 1; // +1 to start after '/'
    size_t end = path.rfind('.');
    std::string fileName = path.substr(start, end - start);
    return fileName;
}



static bool IsPerfectSquare(int num) {
	int root = static_cast<int>(sqrt(num));
	return root*root==num;
}


template <typename T, typename U>
inline U RoundedSqrt(T n) {return static_cast<U>(sqrt(n));}

template <typename T=float>
inline T FloatDiv(T a, T b) {return static_cast<T>(a) / static_cast<T>(b);}

template <typename T>
std::string TupleStr(const std::tuple<T,T,T>& tuple) {

    std::stringstream ss;

    ss<<"(";
    ss<<std::get<0>(tuple)<<","<<std::get<1>(tuple)<<","<<std::get<2>(tuple);
    ss<<")"<<std::endl;
   
    return ss.str();
}

}//autotuning
}//combblas


#endif

