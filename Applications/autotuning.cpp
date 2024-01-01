

#include <cassert>
#include <string>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/Autotuning.h"

#define DEBUG
#define ATIMING

using namespace combblas;

int main(int argc, char ** argv) {
    
    /* ./<binary> <path/to/mat> */
    
    assert(argc>1);
    
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    float dummyBeta = 1.0;
    float dummyAlpha = 0.1;
    int cores = 128;
    int devs = 4;
    autotuning::PlatformParams params(dummyAlpha, dummyBeta, cores, devs);
    autotuning::AutotunerSpGEMM3D tuner(params, autotuning::M_SLURM);
    
    std::string matname(argv[1]);
    
    
    std::shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));
    
    typedef int64_t IT;
    typedef double UT;
    typedef SpDCCols<IT,UT> DER;

    SpParMat<IT,UT,DER> A(grid);
    A.ParallelReadMM(matname, true, maximum<double>());
    SpParMat<IT,UT,DER> B(A);

    
    tuner.Tune(A, B);

    return 0;

}
