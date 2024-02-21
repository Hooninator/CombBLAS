
#ifndef AUTOTUNER_H
#define AUTOTUNER_H


#include "common.h"
#include "SpGEMM3DModel.h"
#include "SpGEMM3DParams.h"
#include "PlatformParams.h"

namespace combblas {


namespace autotuning {


enum TuningMethod {
    BRUTE_FORCE
}typedef TuningMethod;

class Autotuner {

public:
    
    
    /* CONSTRUCTORS */
    

    // Assumes PlatformParams has already been constructed
    Autotuner(PlatformParams& params): platformParams(params) {
        ASSERT(initCalled, "Please call autotuning::Init() first.");
    }
    
    
    //TODO: Need member functions that estimate nnz per proc in 3D grid without actually creating the 3D grid
    //actually creating the grid is likely slow if done lots of times
    //will handle this in SymbolicSpParMat3D

    
    /* TUNING */
    

    /* Main tuning routine for CPU 3DSpGEMM */
    //TODO: Make the tuning method parameter a std::function instance
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams TuneSpGEMM3D(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, TuningMethod method,
                                    std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+".out", autotuning::rank);
#endif

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM3D");
#endif
        
        SpParMat3D<AIT, ANT, ADER> A3D(A, 1, true, false);
        SpParMat3D<BIT, BNT, BDER> B3D(B, 1, false, false);
    
        SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A3D, B3D);
        
        SpGEMM3DParams resultParams; 
        
        switch(method) {
            case BRUTE_FORCE:
            {
                resultParams = SearchBruteForce<SpGEMM3DParams, SpGEMM3DModel>(inputs); 
                break;
            }
            default:
            {
                break;
            }
        }

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM3D");
        infoPtr->PrintGlobal("TuneSpGEMM3D");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }


    /* Main tuning routine for GPU 3DSpGEMM */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMM3DParams TuneSpGEMM3DGPU() {/*TODO*/}

    
    template <typename P, typename M, typename I>
    P SearchBruteForce(I& input) {

#ifdef PROFILE
        infoPtr->StartTimerGlobal("BruteForceSearch");
#endif

        auto searchSpace = P::ConstructSearchSpace(platformParams);
        ASSERT(searchSpace.size()>0, "Search space is of size 0!");

#ifdef PROFILE
        infoPtr->PutGlobal("SearchSpaceSize", std::to_string(searchSpace.size()));
#endif

        P bestParams;  
        M model(platformParams);

        double bestTime = std::numeric_limits<double>::max(); 

        for (P currParams : searchSpace) {


            double currTime = model.EstimateRuntime(input, currParams);
            if (currTime<=bestTime) {
                bestTime = currTime;
                bestParams = currParams;
            }

#ifdef PROFILE
            infoPtr->Put("TotalTime", std::to_string(currTime/1e6));
            infoPtr->WriteInfo();
            infoPtr->Clear();
#endif

        }

#ifdef PROFILE
        infoPtr->EndTimerGlobal("BruteForceSearch");
        infoPtr->PrintGlobal("SearchBruteForce");
#endif
        
        return bestParams;
    }
    

    ~Autotuner(){}

private:
    PlatformParams platformParams;

};//Autotuner


}//autotuning
}//combblas
#endif
