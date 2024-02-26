
#ifndef SPGEMM2DMODELANALYTICAL_H
#define SPGEMM2DMODELANALYTICAL_H


#include "common.h"
#include "SpParMatInfo.h"
#include "CommModel.h"
#include "BcastInfo.h"
#include "LocalSpGEMMModel.h"
#include "MergeModel.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"


namespace combblas {
namespace autotuning {


template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM2DInputs {

public:

    SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpParMat<AIT,ANT,ADER>& A,
                                                SpParMat<BIT,BNT,BDER>& B):
        Ainfo(A),Binfo(B)
    {
    }

    SpParMatInfo<AIT,ANT,ADER> Ainfo;
    SpParMatInfo<BIT,BNT,BDER> Binfo;
};


template <typename D>
class SpGEMM2DModel {
public:

    SpGEMM2DModel(PlatformParams& platformParams ) : platformParams(platformParams)
    {

    }
    

    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpGEMM2DInputs<AIT, ANT, ADER, BIT, BNT, BDER>& inputs,
                                    SpGEMMParams& params) { 
        static_cast<D *>(this)->EstimateRuntimeImpl(inputs, params);
    }

protected:
    PlatformParams platformParams;

};


class SpGEMM2DModelAnalytical : public SpGEMM2DModel<SpGEMM2DModelAnalytical> {
public:


    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntimeImpl(SpGEMM2DInputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, SpGEMMParams& params) {

#ifdef DEBUG
        debugPtr->Log(params.OutStr());
        debugPtr->Print0(params.OutStr());
#endif

#ifdef PROFILE
        infoPtr->Put("Nodes", std::to_string(params.GetNodes()));
        infoPtr->Put("PPN", std::to_string(params.GetPPN()));
        infoPtr->Print("Nodes");
        infoPtr->Print("PPN");
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        // Set dimensions of 3D processor grid
        Ainfo.SetGridDims(params);
        Binfo.SetGridDims(params);

        // Compute nnz per tile in hypothetical 3D grid
        Ainfo.ComputeNnzArr(params);
        Binfo.ComputeNnzArr(params);

        //BROADCAST
        CommModel<AIT> *bcastModel = new PostCommModel<AIT>(platformParams.GetInternodeAlpha(),
                                                    platformParams.GetInternodeBeta(),
                                                     platformParams.GetIntranodeBeta());
        double bcastATime = BcastTime(bcastModel, Ainfo, params, true);
        double bcastBTime = BcastTime(bcastModel, Binfo, params, false);
        
        //LOCAL SpGEMM
        LocalSpGEMMModel<AIT, BIT>* localMultModel = new RooflineLocalSpGEMMModel<AIT, ANT, BIT, BNT>(autotuning::perlmutterParams);
        double localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo, params);

#ifdef PROFILE
        infoPtr->Put("bcastTime-A", std::to_string(bcastATime/1e6));
        infoPtr->Put("bcastTime-B", std::to_string(bcastBTime/1e6));
        infoPtr->Put("multTime", std::to_string(localMultTime/1e6));
#endif

        delete bcastModel;
        delete localMultModel;

        MPI_Barrier(MPI_COMM_WORLD);

        return bcastATime + bcastBTime + localMultTime;
    }


    /* BROADCAST */

    //TODO: Consider nnz estimator class + template to make switching between things here easier
    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpParMatInfo<IT,NT,DER>& Minfo, SpGEMMParams& params, bool row) {

#ifdef PROFILE
        if (row)
            infoPtr->StartTimer("bcastCalcTime-A");
        else
            infoPtr->StartTimer("bcastCalcTime-B");
#endif

        std::vector<IT> * nnz2D = Minfo.GetNnzArr();

        // Compute local bcast times
        std::vector<double> locBcastTimes(params.GetTotalProcs());
        for (int p=0; p<params.GetTotalProcs(); p++) {
            
            // Vector containing nnz for each rank participating in broadcasts with rank p
            std::vector<IT> nnzBcastWorld(params.GetGridDim());
            //TODO: Params class should have methods that return ranks in row/col, then just use std::transform to create bcast world
            if (row) 
                nnzBcastWorld = Minfo.SliceNnzRow(nnz2D, p, params.GetGridDim());
            else
                nnzBcastWorld = Minfo.SliceNnzCol(nnz2D, p, params.GetGridDim());
            
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
        if (row) {
            infoPtr->EndTimer("bcastCalcTime-A");
            infoPtr->Print("bcastCalcTime-A");
        } else {
            infoPtr->EndTimer("bcastCalcTime-B");
            infoPtr->Print("bcastCalcTime-B");
        }
#endif

        return finalTime;
    }


    /* LOCAL SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double LocalMultTime(LocalSpGEMMModel<AIT, BIT>* model, 
                            SpParMatInfo<AIT,ANT,ADER>& Ainfo,
                            SpParMatInfo<BIT,BNT,BDER>& Binfo,
                            SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("multCalcTime");
#endif
        
        auto Adims = Ainfo.GetGridDims(); 
        auto Bdims = Binfo.GetGridDims();

        const int totalProcs = params.GetTotalProcs();

        std::vector<double> * localSpGEMMTimes = new std::vector<double>;
        localSpGEMMTimes->reserve(totalProcs);
        for (int p=0; p<totalProcs; p++) {

            auto ranksA = Ainfo.RowRanks(p, params);
            auto ranksB = Binfo.ColRanks(p, params);

            ASSERT(ranksA.size()==ranksB.size(), "ranksA and ranksB should be the same size, instead got " +
                                            std::to_string(ranksA.size()) +  "," + std::to_string(ranksB.size()));

            for (int i=0; i<ranksA.size(); i++) {
                int rankA = ranksA[i];
                int rankB = ranksB[i];
                LocalSpGEMMInfo<AIT, BIT> * info = new LocalSpGEMMInfo<AIT, BIT> 
                                                    { -1, //placeholder 
                                                    std::get<0>(Adims), std::get<1>(Adims),
                                                    std::get<0>(Bdims), std::get<1>(Bdims),
                                                    Ainfo.GetLocNnzGrid(NNZ_ARR,rankA), 
                                                    Binfo.GetLocNnzGrid(NNZ_ARR,rankB),
                                                    Ainfo.GetGlobDensity(),
                                                    Ainfo.GetLocDensityArr()->at(rankA),
                                                    Binfo.GetGlobDensity(),
                                                    Binfo.GetLocDensityArr()->at(rankB)};
                info->SetFLOPS(params, FLOPS_LOC_DENSITY);
                localSpGEMMTimes->push_back(model->Time(info));
            }

        }


        // Reduce to get max time
        double finalTime = std::reduce(localSpGEMMTimes->begin(),localSpGEMMTimes->end(), 0,
            [](double currMax, double currElem) {
                return std::max(currMax, currElem);
            }
        );

        delete localSpGEMMTimes;

#ifdef PROFILE
        infoPtr->EndTimer("multCalcTime");
        infoPtr->Print("multCalcTime");
#endif

        return finalTime;
    }

    double LayerMergeTime() {
        return 0;
    }
 
};


#ifdef XGB_MODEL
class SpGEMM2DModelXgb : public SpGEMM2DModel<SpGEMM2DModelXgb> {
public:
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntimeImpl(SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, SpGEMMParams& params) {
        return 0;
    }
};
#endif


}//autotuning
}//combblas

#endif





