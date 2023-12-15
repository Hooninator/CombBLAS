


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



using namespace combblas;


int main(int argc, char ** argv) {
    
    int np; int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    assert(argc==2);


    return 0;

}

