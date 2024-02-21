

#include <cassert>
#include <string>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/Autotuning/Autotuner.h"

#define ATIMING

using namespace combblas;

int main(int argc, char ** argv) {
    
    /* ./<binary> <path/to/mat> */
    
    assert(argc>3);
    
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    std::string matpathA(argv[1]);
    std::string matpathB(argv[2]);

    bool permute = (bool)(std::atoi(argv[3]));
    
    autotuning::Init(autotuning::M_SLURM);
    autotuning::Autotuner tuner(autotuning::perlmutterParams);
    
    
    std::shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));
    
    typedef int64_t IT;
    typedef double UT;
    typedef SpDCCols<IT,UT> DER;

    SpParMat<IT,UT,DER> A(grid);
    SpParMat<IT,UT,DER> B(grid);
    A.ParallelReadMM(matpathA, true, maximum<double>());
    B.ParallelReadMM(matpathB, true, maximum<double>());
    if (permute) {
        FullyDistVec<IT,UT> p(A.getcommgrid());
        p.iota(A.getnrow(), 0);
        p.RandPerm();
        (B)(p,p,true);
        matpathB += std::string("-permute");
    }

    
    auto resultParams = tuner.TuneSpGEMM2D(A, B, autotuning::BRUTE_FORCE, matpathA, matpathB);
    
    autotuning::Finalize();

    return 0;

}
