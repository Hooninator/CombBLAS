
#ifndef SPGEMM3DMODEL_H
#define SPGEMM3DMODEL_H


#include "common.h"

#include "SpGEMM3DMatrixInfo.h"

#include "CommModel.h"
#include "BcastInfo.h"
#include "LocalSpGEMMModel.h"
#include "MergeModel.h"
#include "SpGEMM3DParams.h"

#include "PlatformParams.h"


namespace combblas {
namespace autotuning {


template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM3DInputs {

public:

    SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo,
                                                SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo):
    Ainfo(Ainfo), Binfo(Binfo) {}

    SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo;
    SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo;
};


/* Output of tune routine. Relevant SpGEMM3D params that should be used to create a CommGrid3D 
 * Also defines runtime estimation model. TODO: Move that functionality to a dedicated model class
 */
class SpGEMM3DModel {
public:


    SpGEMM3DModel(PlatformParams& platformParams ) : platformParams(platformParams)
    {

    }



    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpGEMM3DInputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, SpGEMM3DParams& params) {

#ifdef DEBUG
        debugPtr->Log(params.OutStr());
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        CommModel<AIT> *bcastModel = new PostCommModel<AIT>(platformParams.GetInternodeAlpha(),
                                                    platformParams.GetInternodeBeta(),
                                                     platformParams.GetIntranodeBeta());
        double bcastATime = BcastTime(bcastModel, Ainfo, params);
        double bcastBTime = BcastTime(bcastModel, Binfo, params);
        
        LocalSpGEMMModel * localMultModel = new RegressionLocalSpGEMMModel(autotuning::regSpGEMMPerlmutter);
        double localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo, params);

#ifdef PROFILE
        statPtr->Log("BcastA time " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Log("BcastB time " + std::to_string(bcastBTime/1e6) + "s");
        statPtr->Log("LocalMult time " + std::to_string(localMultTime/1e6) + "s");
#endif
        delete bcastModel;
        delete localMultModel;

        return 0;
    }


    /* BROADCAST */

    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, SpGEMM3DParams& params) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        IT localNnzApprox = Minfo.ComputeLocalNnz(params.GetPPN(), params.GetNodes(), params.GetLayers());
        IT msgSize = Minfo.ComputeMsgSize(localNnzApprox);

        CommOpts * opts = new CommOpts{
            //gridSize <= params.GetCoresPerNode() ? true : false //intranode
            false
        };
        CommInfo<IT> * info = MakeBcastCommInfo(params.GetRowSize(), params.GetTotalProcs(), msgSize); 

        double singleBcastTime = bcastModel->ComputeTime(info, opts);
        double finalTime = singleBcastTime * sqrt(params.GetGridSize());

        delete info;
        delete opts;
        
#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
	    statPtr->Log("Local nnz estimate: " + std::to_string(localNnzApprox));
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
#endif

        return finalTime;
    }


    /* LOCAL SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double LocalMultTime(LocalSpGEMMModel * model, 
                            SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo,
                            SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo,
                            SpGEMM3DParams& params) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        long long localFLOPS = model->ApproxLocalMultFLOPSDensity(Ainfo, Binfo, params.GetTotalProcs(), params.GetGridSize());
        LocalSpGEMMInfo * info = new LocalSpGEMMInfo { localFLOPS };

        double finalTime = model->ComputeTime(info);

        delete info;

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("LocalMult calc time " + std::to_string(t1) + "s");
#endif

        return finalTime;
    }

    double LayerMergeTime() {
        return 0;
    }
 
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;}
    

private:
    
    PlatformParams platformParams;

};



}//autotuning
}//combblas

#endif





