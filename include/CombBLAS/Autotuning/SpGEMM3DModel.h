
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

    SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpParMat3D<AIT,ANT,ADER>& A,
                                                SpParMat3D<BIT,BNT,BDER>& B):
        Ainfo(A),Binfo(B)
    {
    }

    SpGEMM3DMatrixInfo<AIT,ANT,ADER> Ainfo;
    SpGEMM3DMatrixInfo<BIT,BNT,BDER> Binfo;
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

        // Compute nnz per tile in hypothetical 3D grid
        Ainfo.SetNnzArr(params.GetPPN(), params.GetNodes(), params.GetLayers());
        Binfo.SetNnzArr(params.GetPPN(), params.GetNodes(), params.GetLayers());


        //BROADCAST
        CommModel<AIT> *bcastModel = new PostCommModel<AIT>(platformParams.GetInternodeAlpha(),
                                                    platformParams.GetInternodeBeta(),
                                                     platformParams.GetIntranodeBeta());
        double bcastATime = BcastTime(bcastModel, Ainfo, params, true);
        double bcastBTime = BcastTime(bcastModel, Binfo, params, false);
        
        //LOCAL SpGEMM
        LocalSpGEMMModel<AIT>* localMultModel = new RooflineLocalSpGEMMModel<AIT, ANT>(autotuning::perlmutterParams);
        double localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo, params);

#ifdef PROFILE
        statPtr->Log("BcastA time " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Log("BcastB time " + std::to_string(bcastBTime/1e6) + "s");
        statPtr->Log("LocalMult time " + std::to_string(localMultTime/1e6) + "s");
#endif
        delete bcastModel;
        delete localMultModel;

        MPI_Barrier(MPI_COMM_WORLD);

        return 0;
    }


    /* BROADCAST */

    //TODO: Consider nnz estimator class + template to make switching between things here easier
    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, SpGEMM3DParams& params, bool row) {

#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif

        std::vector<IT> * nnz3D = Minfo.GetNnzArr();

#ifdef DEBUG
        debugPtr->Log("nnz arr size:" + std::to_string(nnz3D->size()));
#endif

        if (params.GetGridSize()==1) return 0; //no bcasts in this case
        
        
        // Compute local bcast times
        std::vector<double> locBcastTimes(params.GetTotalProcs());
        for (int p=0; p<params.GetTotalProcs(); p++) {
            
            // Vector containing nnz for each rank participating in broadcasts with rank p
            std::vector<IT> nnzBcastWorld(params.GetGridDim());
            if (row) 
                nnzBcastWorld = Minfo.SliceNnzRow(nnz3D, p, params.GetGridDim());
            else
                nnzBcastWorld = Minfo.SliceNnzCol(nnz3D, p, params.GetGridDim());
            
            // Compute and sum all times for all bcasts rank p participates in 
            double locBcastTime = std::reduce(nnzBcastWorld.begin(), nnzBcastWorld.end(), 0, 
                [&Minfo, &bcastModel, &params](double sum, IT nnz) {
                    IT msgSize = Minfo.ComputeMsgSize(nnz);

                    CommOpts * opts = new CommOpts{
                        //gridSize <= params.GetCoresPerNode() ? true : false //intranode
                        false
                    };

                    CommInfo<IT> * info = MakeBcastCommInfo(params.GetGridDim(),  msgSize); 

                    double singleBcastTime = bcastModel->Time(info, opts);

                    delete info;
                    delete opts;

                    return singleBcastTime + sum;
                }
            );
            
            locBcastTimes[p] = locBcastTime;

        }

        // Reduce to get max time
        double finalTime = std::reduce(locBcastTimes.begin(), locBcastTimes.end(), 0,
            [](double currMax, double currElem) {
                return std::max(currMax, currElem);
            }
        );

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
        statPtr->Print("Bcast calc time " + std::to_string(t1) + "s");
#endif

        return finalTime;
    }


    /* LOCAL SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double LocalMultTime(LocalSpGEMMModel<AIT>* model, 
                            SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo,
                            SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo,
                            SpGEMM3DParams& params) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        long long localFLOPS = model->ApproxLocalMultFLOPSDensity(Ainfo, Binfo, params.GetTotalProcs(), params.GetGridSize());
        
        auto Adims3D = Ainfo.ComputeLocDims3D(params.GetPPN(), params.GetNodes(), params.GetLayers());
        auto Bdims3D = Binfo.ComputeLocDims3D(params.GetPPN(), params.GetNodes(), params.GetLayers());

        const int totalProcs = params.GetTotalProcs();

        double finalTime = 0;
        for (int p=0; p<totalProcs; p++) {
            LocalSpGEMMInfo<AIT> * info = new LocalSpGEMMInfo<AIT> { localFLOPS, 
                                                            std::get<0>(Adims3D), std::get<1>(Adims3D),
                                                            std::get<0>(Bdims3D), std::get<1>(Bdims3D),
                                                            Ainfo.GetNnzArr()->at(p), 
                                                            Binfo.GetNnzArr()->at(p)};
            double localTime = model->Time(info);
            finalTime+=localTime;

            delete info;
        }

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





