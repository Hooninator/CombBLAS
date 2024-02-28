
#ifndef AUTOTUNER_H
#define AUTOTUNER_H


#include "common.h"
#include "SpGEMM2DModel.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"

namespace combblas {


namespace autotuning {

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
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DAnalytical(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                    std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+".out", autotuning::rank);
#endif

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DAnalytical");
#endif

        typedef SpGEMM2DModel<SpGEMM2DModelAnalytical> ModelType;
        
        SpGEMM2DModelAnalytical::Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A, B);
        
        SpGEMMParams resultParams; 
        resultParams = SearchBruteForce<SpGEMMParams, ModelType>(inputs);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DAnalytical");
        infoPtr->PrintGlobal("TuneSpGEMM2DAnalytical");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    SpGEMMParams TuneSpGEMM2DXgb(SpParMat<AIT, ANT, ADER>& A, SpParMat<BIT, BNT, BDER>& B, 
                                    std::string& matpathA, std::string& matpathB){

#ifdef PROFILE
        std::string matnameA = ExtractMatName(matpathA);
        std::string matnameB = ExtractMatName(matpathA);
        infoPtr = new InfoLog("info-"+matnameA+"x"+matnameB+".out", autotuning::rank);
#endif

#ifdef PROFILE
        infoPtr->StartTimerGlobal("TuneSpGEMM2DXgb");
#endif

        typedef SpGEMM2DModel<SpGEMM2DModelXgb> ModelType;
        
        SpGEMM2DModelXgb::Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs(A, B);
        
        SpGEMMParams resultParams; 
        resultParams = SearchInference<SpGEMMParams, ModelType>(inputs);

#ifdef PROFILE
        infoPtr->EndTimerGlobal("TuneSpGEMM2DXgb");
        infoPtr->PrintGlobal("TuneSpGEMM2DXgb");
#endif

#ifdef PROFILE
        infoPtr->WriteInfoGlobal();
        delete infoPtr;
#endif

        return resultParams;

    }


    
    template <typename P, typename M, typename I>
    P SearchBruteForce(I& inputs) {

#ifdef PROFILE
        infoPtr->StartTimerGlobal("BruteForceSearch");
#endif

        //TODO: This makes this routine not generic since not all problems will have a 'searchspace2d' function
        auto searchSpace = P::ConstructSearchSpace2D(platformParams);
        ASSERT(searchSpace.size()>0, "Search space is of size 0!");

#ifdef PROFILE
        infoPtr->PutGlobal("SearchSpaceSize", std::to_string(searchSpace.size()));
#endif

        P bestParams;  
        M model(platformParams);

        double bestTime = std::numeric_limits<double>::max(); 

        for (P currParams : searchSpace) {

            double currTime = model.EstimateRuntime(inputs, currParams);
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
        infoPtr->PrintGlobal("BruteForceSearch");
#endif
        
        return bestParams;
    }


    template <typename P, typename M, typename I>
    P SearchInference(I& inputs) {
        
#ifdef PROFILE
        infoPtr->StartTimerGlobal("InferenceSearch");
#endif

        std::vector<P> searchSpace = P::ConstructSearchSpace2D(platformParams);

        M model(platformParams);

        //NOTE: FeatureMat is in row-major order
        std::vector<float> featureMat;
        featureMat = model.MakeFeatureMat(inputs, searchSpace);

        std::vector<float> predictions;
        predictions = model.Predict(featureMat);

        auto minElem = std::min_element(predictions.begin(), predictions.end());
        int minIdx = std::distance(predictions.begin(), minElem);

        P bestParams = searchSpace[minIdx];

#ifdef PROFILE
        infoPtr->PutGlobal("BestParams", bestParams.OutStr());
        infoPtr->EndTimerGlobal("InferenceSearch");
        infoPtr->PrintGlobal("InferenceSearch");
        infoPtr->PrintGlobal("BestParams");
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
