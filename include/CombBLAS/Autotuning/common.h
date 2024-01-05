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

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string(message)); \
        } \
    } while (false)



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

}//autotuning
}//combblas


#endif

