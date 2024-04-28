

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

int main(int argc, char ** argv) {
    
    //TODO: Make actual argparser
    /* ./<binary> <path/to/matA> <path/to/matB> <permute> <maxnodes> <domult>*/
    
    assert(argc>4);
    
    int rank; int n;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    {
        autotuning::Init(autotuning::M_SLURM);
        autotuning::Autotuner tuner(autotuning::perlmutterParams);

        std::string matpathA(argv[1]);
        std::string matpathB(argv[2]);

        bool permute = (bool)(std::atoi(argv[3]));
        
        
        std::shared_ptr<CommGrid> grid;
        grid.reset(new CommGrid(MPI_COMM_WORLD,0,0));
        
        typedef int64_t IT;
        typedef double UT;
        typedef SpDCCols<IT,UT> DER;
        typedef PlusTimesSRing<UT,UT> PTTF;

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
    
        double SpGEMMTime = 0;
        double tunedSpGEMMTime = 0;
        double redistTime = 0;
        double tuningTime = 0;
    
        double stime, etime;
    
        stime = MPI_Wtime();
        Mult_AnXBn_Synch<PTTF, UT, DER>(A,B,false,false);
        etime = MPI_Wtime();
    
        SpGEMMTime += (etime - stime);
    
        int maxNodes = std::atoi(argv[4]);
        
        stime = MPI_Wtime();
    
        autotuning::SpGEMMParams resultParams;
        resultParams = tuner.TuneSpGEMM2DAnalytical(A,B,matpathA,matpathB,maxNodes);
    
        etime = MPI_Wtime();
        tuningTime += (etime - stime);
    
        bool doMult = (bool)(std::atoi(argv[5]));
    
        if (doMult) {
    
            auto tunedGrid = resultParams.MakeGridFromParams();
    
            if (tunedGrid!=NULL) {
                stime = MPI_Wtime();
                SpParMat<IT,UT,DER> ATuned(A.seqptr(), tunedGrid);
                SpParMat<IT,UT,DER> BTuned(B.seqptr(), tunedGrid);
                etime = MPI_Wtime();
    
                redistTime = (etime - stime);
    
                stime = MPI_Wtime();
                Mult_AnXBn_Synch<PTTF, UT, DER>(ATuned, BTuned);
                etime = MPI_Wtime();
    
                tunedSpGEMMTime += (etime - stime);
            }
    
            if (rank==0) {
                std::cout<<"SpGEMM Time: "<<SpGEMMTime<<std::endl;
                std::cout<<"Tuned SpGEMM Time: "<<tunedSpGEMMTime<<std::endl;
                std::cout<<"Tuning Time: "<<tuningTime<<std::endl;
                std::cout<<"Redistribution Time: "<<redistTime<<std::endl;
            }
    
        }
        
        autotuning::Finalize();
    }
    
    MPI_Finalize();
    
    return 0;

}
