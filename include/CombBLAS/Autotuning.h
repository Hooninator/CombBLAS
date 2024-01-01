


#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>

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



namespace combblas {


template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class AutotunerSpGEMM3D {

public:
    
    typedef SpParMat<AIT, ANT, ADER> AMat;
    typedef SpParMat<BIT, BNT, BDER> BMat;

    struct SpGEMM3DParams {
        int nNodes;
        int ppn;
        int layers; 
    } typedef SpGEMM3DParams;
    
    
    AutotunerSpGEMM3D(PlatformParams& params, AMat& A, BMat&  B):
    params(params), A(A), B(B), grid2D(A.getcommgrid())
    {

    }
    
    
    CommGrid3D tune(){/*TODO*/}
    
    
    double ABcastTime(){return 0;}
    double BBcastTime(){return 0;}
    double LocalMultTime(){return 0;}
    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;} 
    
    
    void GetSlurmInfo(){/*TODO*/}
    

    ~AutotunerSpGEMM3D(){}

private:
    PlatformParams params;
    AMat A; BMat B;
    std::shared_ptr<CommGrid> grid2D;

};


}//combblas
