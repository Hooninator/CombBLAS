

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iterator>
#include <cassert>
#include <cstdlib>



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



namespace combblas {

namespace autotuning {

template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SymbolicSpParMat3D {

};


}//autotuning 
}//combblas
