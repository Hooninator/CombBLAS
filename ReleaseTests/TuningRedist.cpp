

#include <cassert>
#include <string>

#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/SpParMat3D.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/Autotuning/Autotuner.h"

#define ATIMING
#define THREADED

using namespace combblas;

unsigned int NextSmallestPerfectSquare(unsigned int n) {
    // Find the square root of the input number
    unsigned int root = static_cast<unsigned int>(std::sqrt(n));
    
    // If the square root is not a perfect square, add 1 to it
    if (std::sqrt(root) != std::ceil(std::sqrt(root))) {
        root++;
    }
    
    // Return the square of the next integer
    return root * root;
}

template <typename IT, typename UT,typename  DER>
void RunRedistTest(SpParMat<IT,UT,DER>& A, std::string matpathA, int rank)
{

    autotuning::SpGEMMParams defaultParams = autotuning::SpGEMMParams::GetDefaultParams();
    autotuning::SpGEMMParams scalingDownParams(1, NextSmallestPerfectSquare(defaultParams.GetPPN()-1), 1);

    auto grid = A.getcommgrid();

    /**** TEST SCALING DOWN ****/
    /////////////////////////////

    if (rank==0)
        std::cout<<"Testing scaling down..."<<std::endl;

    A.ParallelWriteMM("./stomach-correct.mtx", true);

    // Re-distribute from default to smaller process grid
    DER * AdownLocalRedist = scalingDownParams.ReDistributeSpMat(A.seqptr(), defaultParams);

    auto smallGrid = scalingDownParams.MakeGridFromParams();
    if (smallGrid!=NULL) 
    {
        DER * AdownLocalRedistCpy = new DER(*AdownLocalRedist);
        SpParMat<IT,UT,DER> AdownRedist(AdownLocalRedistCpy, smallGrid);
        AdownRedist.ParallelWriteMM("./stomach-down.mtx", true );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank==0)
        std::cout<<"Scaling down done"<<std::endl;

    ///////////////////////////
    /**** END TEST SCALING DOWN ****/
   
    MPI_Barrier(MPI_COMM_WORLD);

    /**** TEST SCALING UP ****/
    ///////////////////////////
    if (rank==0)
        std::cout<<"Testing scaling up..."<<std::endl;

    // Re-redistribute from smaller to default process grid
    DER * AupLocalRedist = defaultParams.ReDistributeSpMat(AdownLocalRedist, scalingDownParams);


    if (grid!=NULL)
    {
        DER * AupLocalRedistCpy = new DER(*AupLocalRedist);
        SpParMat<IT,UT,DER> AupRedist(AupLocalRedistCpy, grid);
        AupRedist.ParallelWriteMM("./stomach-up.mtx", true);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank==0)
        std::cout<<"Scaling up done"<<std::endl;

    /**** END TEST SCALING UP ****/
    ///////////////////////////
    
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char ** argv) 
{
    
    /* ./<binary> <path/to/mat> */
    
    assert(argc>1);
    
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    {
        autotuning::Init(autotuning::M_OMPI);
        autotuning::Autotuner tuner(autotuning::fractusParams);

        std::string matpathA(argv[1]);

        std::shared_ptr<CommGrid> grid;
        grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));
        
        typedef int64_t IT;
        typedef double UT;
        typedef SpDCCols<IT,UT> DER;
        typedef PlusTimesSRing<UT,UT> PTTF;

        SpParMat<IT,UT,DER> A(grid);
        A.ParallelReadMM(matpathA, true, maximum<double>());

        RunRedistTest(A, matpathA, rank);

    }

    MPI_Finalize();

}
