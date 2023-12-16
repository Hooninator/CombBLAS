


#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cassert>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "gspgemm/meSpGEMM.h"

#define VERBOSE

#ifdef VERBOSE
#define PRINT(msg) std::cout<<msg<<std::endl;
#else
#define PRINT(msg)
#endif

#define ITERS 10

#ifdef TIMING
double cblas_alltoalltime;
double cblas_allgathertime;
//////////////////////////
double mcl_Abcasttime;
double mcl_Bbcasttime;
double mcl_localspgemmtime;
double mcl_multiwaymergetime;
double mcl_kselecttime;
double mcl_prunecolumntime;
double mcl_symbolictime;
double mcl_totaltime;
double mcl_tt;
int64_t mcl_nnzc;
///////////////////////////
double mcl_Abcasttime_prev;
double mcl_Bbcasttime_prev;
double mcl_localspgemmtime_prev;
double mcl_multiwaymergetime_prev;
double mcl_kselecttime_prev;
double mcl_prunecolumntime_prev;
double mcl_symbolictime_prev;
double mcl_totaltime_prev;
double mcl_tt_prev;
int64_t mcl_nnzc_prev;
#endif


using namespace combblas;


void run2DGPU(int argc, char ** argv) {

    int np; int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* Setup comm grid and sparse matrix */
    std::string matName(argv[2]);
    SpParHelper::Print("Matrix name: "+matName+"\n");
    
    shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));

    /* Useful types */
    typedef double NT;
    typedef int64_t IT;
    typedef SpDCCols <IT, NT> DER;
    
    SpParMat<IT, NT, DER> A(grid);
    
    /* Read in matrix */
    A.ParallelReadMM(matName, true, maximum<NT>());
    
    double loadBalance = A.LoadImbalance();
    SpParHelper::Print("Load balance: " + std::to_string(loadBalance) + "\n");

    /* Matrix B, randomly permute columns */
    SpParMat<IT, NT, DER> B(A);
    FullyDistVec<IT, NT> p(A.getcommgrid());
    p.iota(A.getnrow(), 0);
    p.RandPerm();
    (B)(p,p,true);
    
    /* Normal fmadd semiring */
    typedef PlusTimesSRing<double,double> PTTF;
    
    
    double totalTime=0.0;
    int64_t perProcessMem = (256*(pow(10,9))) / np;
   
    for (int i=0; i<ITERS; i++) {
        
        SpParHelper::Print("Iteration " + std::to_string(i)+"\n");
        
        auto stime = MPI_Wtime(); 
    
        MemEfficientSpGEMMg<PTTF, NT, DER>(
                            A, B, //A, B
                            1, //phases
                            (NT)(1.0/10000.0), //hardThreshold
                            (IT)1100, //selectNum
                            (IT)1400, //recoverNum
                            (NT)0.9, //recoverPct
                            0, //kselectVersion
                            perProcessMem, //perProcessMem
                            LSPG_NSPARSE, //local_spgemm
                            1); //nrounds
        
        auto etime = MPI_Wtime();
        totalTime += (etime-stime);
    
    } 
    
    double avgTime = totalTime / ITERS;
    
    SpParHelper::Print("Avg time: " + std::to_string(avgTime) + "\n");

}


void run2DCPU(int argc, char ** argv) {
    //TODO
}

void run3DCPU(int argc, char ** argv) {
    //TODO
}

int main(int argc, char ** argv) {
    
    assert(argc>2);
    std::string algorithm(argv[1]);
    
    if (algorithm.compare("2DGPU")==0) {
        run2DGPU(argc, argv);
    } else {
        std::cerr<<"Algorithm "<<algorithm<<" not recognized/supported"<<std::endl;
        exit(1);
    }
    
    
    return 0;

}

