
#ifndef SPGEMM2DMODEL_H
#define SPGEMM2DMODEL_H


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

template <typename AIT, typename ANT,typename ADER,typename BIT,typename BNT,typename BDER>
class SpGEMM2DInputs {

public:

    SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER>()
    {
    }

};

template <typename MT>
class SpGEMM2DModel {
public:

    SpGEMM2DModel(){}

    void Create(PlatformParams& params)
    {
        this->platformParams = params;
        static_cast<MT*>(this)->CreateImpl();
    }


    /* Get runtime estimate of a certain combo of parameters */
    template <typename I>
    std::vector<float> Predict(I& inputs, std::vector<SpGEMMParams>& params) { 
        return static_cast<MT*>(this)->PredictImpl(inputs, params);
    }

    std::vector<float> Predict(std::vector<float>& X) {
        return static_cast<MT*>(this)->PredictImpl(X);
    }

    //TODO: This should be able to return vectors of things other than floats
    template <typename I>
    std::vector<float> MakeFeatureMat(I& inputs, std::vector<SpGEMMParams>& searchSpace) {
        return static_cast<MT*>(this)->MakeFeatureMatImpl(inputs,searchSpace);
    }

    template <typename I>
    std::vector<float> MakeFeatureMat(I& inputs, SpGEMMParams& params) {
        return static_cast<MT*>(this)->MakeFeatureMatImpl(inputs,params);
    }

    //TODO: replace this with somethine non-embarrassing 
#ifdef PROFILE
    void WritePrediction(std::vector<SpGEMMParams>& searchSpace, std::vector<float>& predictions) {
        infoPtr->OFS()<<"----RUNTIME ESTIMATES----"<<std::endl;
        ASSERT(searchSpace.size()==predictions.size(), "sizes not equal");
        for (int i=0; i<searchSpace.size(); i++) {
            infoPtr->OFS()<<searchSpace[i]<<":"<<predictions[i]/1e6<<"s ";
        }
        infoPtr->OFS()<<std::endl;
    }
#endif

protected:

    PlatformParams platformParams;

};


class SpGEMM2DModelAnalytical : public SpGEMM2DModel<SpGEMM2DModelAnalytical> {
public:

    void CreateImpl() {}
    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoAnalytical : public SpParMatInfo<IT,NT,DER> {
    public:
        
        /* (row,col,nnz) */
        //TODO: For col split, no need to store row idx, and for row split, no need to store col idx
        typedef std::vector<std::tuple<IT,IT,IT>> NnzTuples;

        using SpParMatInfo<IT,NT,DER>::SpParMatInfo; 
        using SpParMatInfo<IT,NT,DER>::locNnz; 
        using SpParMatInfo<IT,NT,DER>::locNcolsExact; 
        using SpParMatInfo<IT,NT,DER>::locNrowsExact; 
        using SpParMatInfo<IT,NT,DER>::locNcols; 
        using SpParMatInfo<IT,NT,DER>::locNrows; 
        using SpParMatInfo<IT,NT,DER>::locMat; 
        using SpParMatInfo<IT,NT,DER>::split; 
        using SpParMatInfo<IT,NT,DER>::rowRank; 
        using SpParMatInfo<IT,NT,DER>::colRank; 
        using SpParMatInfo<IT,NT,DER>::gridDims; 
        using SpParMatInfo<IT,NT,DER>::globDensity; 

        
        SpParMatInfoAnalytical(SpParMat<IT,NT,DER>& Mat): 
            SpParMatInfo<IT,NT,DER>(Mat),
            nnzArr(new std::vector<IT>(0)),
            locDensityArr(new std::vector<float>(worldSize))
        {
            
            locDensityArr->insert(locDensityArr->begin() + rank,
                                    static_cast<float>(locNnz) / static_cast<float>(locNcolsExact*locNrowsExact));
            MPI_Allgather(MPI_IN_PLACE, 1, MPI_FLOAT, (void*)(locDensityArr->data()), 1, MPI_FLOAT, MPI_COMM_WORLD);

            split = COL_SPLIT; // This is much nicer, and in 2d it doesn't matter

#ifdef NNZ_TUPLES_COL

#ifdef PROFILE
            infoPtr->StartTimer("nnzTuplesColInit");
#endif
            nnzTuples = NnzTuplesCol();
#ifdef PROFILE
            infoPtr->EndTimer("nnzTuplesColInit");
#endif

#endif
            MPI_Barrier(MPI_COMM_WORLD);

        }

        /* Create array of tuples containing nnz per tile column for this processor's local tile  */
        NnzTuples * NnzTuplesCol() {

#ifdef PROFILE
            infoPtr->StartTimer("locNnzTuplesColInit");
#endif

            auto _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
            _nnzTuples->reserve(locNcolsExact);

            // Init local data
            int locTupleSize = 0;
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                if (colIter.nnz()>NNZ_THRESH) {
                    _nnzTuples->push_back( std::tuple<IT,IT,IT>{colRank,  colIter.colid() + locNcols*rowRank, colIter.nnz()} );
                }
            }

#ifdef PROFILE
            infoPtr->EndTimer("locNnzTuplesColInit");
#endif

#ifdef DEBUG
            debugPtr->Log("locNnzTuples col");
            for (int i=0; i<_nnzTuples->size(); i++) {
                debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
            }
#endif

            return _nnzTuples;

        }



        /* Initialize array of tuples containing nnz per tile row on this processor's local tile */
        NnzTuples * NnzTuplesRow() {

#ifdef PROFILE
            infoPtr->StartTimer("locNnzTuplesRowInit");
#endif

            // JB: I can't figure out a way to avoid mutating nnz during iteration, so we can't just use std::tuple
            std::map<std::tuple<IT,IT>, IT> nnzMap;
            for (auto colIter = locMat->begcol(); colIter != locMat->endcol(); colIter++) {
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    std::tuple<IT,IT> t{nzIter.rowid() + locNrows*colRank, rowRank};
                    nnzMap.emplace(t, 0);
                    nnzMap[t] += 1;
                }
            }


            auto  _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
            _nnzTuples->reserve(locNrowsExact);

            std::for_each(nnzMap.begin(), nnzMap.end(),
                [&_nnzTuples](auto& elem)  {
                    std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
                    _nnzTuples->push_back( t );
                }
            );

#ifdef PROFILE
            infoPtr->EndTimer("locNnzTuplesRowInit");
#endif

#ifdef DEBUG
            debugPtr->Log("locNnzTuples row");
            for (int i=0; i<_nnzTuples->size(); i++) {
                debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
            }
#endif

            return _nnzTuples;

        }


        /* Approximate local nnz using matrix globDensity
         * This actually just computes the avg nnz per processor
         */
        IT ComputeLocNnzGlobDensity() {

            IT localNcols = std::get<1>(gridDims);
            IT localNrows = std::get<0>(gridDims);
            IT localMatSize = localNcols * localNrows;

            IT localNnzApprox = static_cast<IT>(globDensity * localMatSize);
            return localNnzApprox;
        }


        /* Approximate local nnz using matrix locDensityArr
         */
        IT ComputeLocNnzLocDensity(int procRank) {

            IT localNcols = std::get<1>(gridDims);
            IT localNrows = std::get<0>(gridDims);
            IT localMatSize = localNcols * localNrows;

            IT localNnzApprox = static_cast<IT>(locDensityArr->at(procRank) * localMatSize);
            return localNnzApprox;
        }


        void ComputeNnzArr(SpGEMMParams& params) {

#ifdef PROFILE
            infoPtr->StartTimer("ComputeNnzArr");
#endif

            nnzArr->clear();
            nnzArr->resize(params.GetTotalProcs());

            switch(split) {
                case COL_SPLIT:
                {
                    ComputeNnzArrColSplit(params);
                    break;
                }
                case ROW_SPLIT:
                {
                    ComputeNnzArrRowSplit(params);
                    break;
                }
                default:
                {
                    UNREACH_ERR();
                }
            }

#ifdef PROFILE
            infoPtr->EndTimer("ComputeNnzArr");
#endif

        }

        
        /* Given local nnz in initial 2D processor grid, compute nnz per processor in 3D processr grid
         * WITHOUT explicitly forming the 3D processor grid. */
        void ComputeNnzArrColSplit(SpGEMMParams& params) {

            const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_COL
            // Local nnz array
            std::for_each(nnzTuples->begin(), nnzTuples->end(),
                [&params,this](auto& t) {
                    int i = std::get<0>(t);
                    int j = std::get<1>(t);
                    int owner = ComputeOwnerGrid(params, i*this->locNrows, j, COL_SPLIT);
                    this->nnzArr->at(owner) += std::get<2>(t);
                }
            );
#else
            // Just use local matrix
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                int j = colIter.colid();
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    int i = nzIter.rowid();
                    int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), COL_SPLIT);
                    nnzArr->at(owner) += 1;
                }
            }
#endif

            // Allreduce to get complete counts for each process
            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
         debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr A: "});
#endif

        }
        


        void ComputeNnzArrRowSplit(SpGEMMParams& params) {

            const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_ROW
            // Local data
            std::for_each(nnzTuples->begin(), nnzTuples->end(),
                [&params, this](auto& t) {
                    int i = std::get<0>(t);
                    int j = std::get<1>(t);
                    int owner = ComputeOwnerGrid(params, i, j*this->locNcols, ROW_SPLIT);
                    this->nnzArr->at(owner) += std::get<2>(t);
                }
            );
#else
            for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
                int j = colIter.colid();
                for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                    int i = nzIter.rowid();
                    int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), ROW_SPLIT);
                    nnzArr->at(owner) += 1;
                }
            }
#endif

            // Allreduce to sum all nnz
            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
            debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr B: "});
#endif

        }
        
        IT ComputeLocNnzGrid(NNZ_STRAT strat, int procRank) {
            switch(strat) {
                case NNZ_GLOB_DENSITY:
                    return ComputeLocNnzGlobDensity();
                case NNZ_LOC_DENSITY:
                    return ComputeLocNnzLocDensity(procRank);
                case NNZ_ARR:
                    return nnzArr->at(procRank);
                default:
                    UNREACH_ERR();
            }
            return 0;
        }

    
        int ComputeOwnerGrid(SpGEMMParams& params, const int i, const int j, SPLIT split) {

            const int layers = params.GetLayers();
            const int gridDim = params.GetGridDim();
            const int gridSize = params.GetGridSize();

            IT locNrowsGrid = std::get<0>(gridDims);
            IT locNcolsGrid = std::get<1>(gridDims);

            IT colDiv;
            IT rowDiv;
            IT layerDiv;

            int layerIdx;

            if (split==COL_SPLIT) {
                colDiv = locNcolsGrid*layers;
                rowDiv = locNrowsGrid;
                layerDiv = locNcolsGrid;
                layerIdx = j;
            } else if (split==ROW_SPLIT) {
                colDiv = locNcolsGrid;
                rowDiv = locNrowsGrid*layers;
                layerDiv = locNrowsGrid;
                layerIdx = i;
            }

            const int prow = std::min(static_cast<IT>(i / rowDiv), static_cast<IT>(gridDim-1));
            const int pcol = std::min(static_cast<IT>(j / colDiv), static_cast<IT>(gridDim-1));
            const int player = std::min(static_cast<IT>((layerIdx / layerDiv)%layers), static_cast<IT>(layers-1));

            return (pcol + prow*gridDim + player*gridSize);
        }
    
        
        /* Sum nnz in procRank's row of the hypothetical 3D grid */
        std::vector<IT> SliceNnzRow(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
            return std::vector<IT>(nnzArr->begin()+(procRank/gridDim), nnzArr->begin()+(procRank/gridDim)+gridDim);
        }


        /* Sum nnz in procRank's column of hypothetical 3D grid */
        std::vector<IT> SliceNnzCol(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
            //TODO: Can we use C++17 algorithms for this?
            std::vector<IT> result(gridDim);
            for (int p=0; p<gridDim; p++) {
                result[p] = nnzArr->at((procRank%gridDim)+p*gridDim);
            }
            return result;
        }


        inline std::vector<IT> * GetNnzArr() {return nnzArr;}
        inline std::vector<float> * GetLocDensityArr() const {return locDensityArr;}
        
    private:

        std::vector<float> * locDensityArr;
        NnzTuples * nnzTuples;

        // Stores nnz per processor in hypothetical 3D grid
        std::vector<IT> * nnzArr;
        

    };


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {

    public:

        Inputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpParMat<AIT,ANT,ADER>& A,
                                                    SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A),Binfo(B)
        {
        }

        SpParMatInfoAnalytical<AIT,ANT,ADER> Ainfo;
        SpParMatInfoAnalytical<BIT,BNT,BDER> Binfo;
    };


    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> PredictImpl(Inputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, std::vector<SpGEMMParams>& searchSpace) {
        
        std::vector<float> predictions;
        predictions.reserve(searchSpace.size());
        for (auto params : searchSpace) {
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
            float bcastATime = BcastTime(bcastModel, Ainfo, params, true);
            float bcastBTime = BcastTime(bcastModel, Binfo, params, false);
            
            //LOCAL SpGEMM
            LocalSpGEMMModel<AIT, BIT>* localMultModel = new RooflineLocalSpGEMMModel<AIT, ANT, BIT, BNT>(autotuning::perlmutterParams);
            float localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo, params);

#ifdef PROFILE
            infoPtr->Put("bcastTime-A", std::to_string(bcastATime/1e6));
            infoPtr->Put("bcastTime-B", std::to_string(bcastBTime/1e6));
            infoPtr->Put("multTime", std::to_string(localMultTime/1e6));
#endif

            delete bcastModel;
            delete localMultModel;

            MPI_Barrier(MPI_COMM_WORLD);

            float time =  bcastATime + bcastBTime + localMultTime;

#ifdef PROFILE
            infoPtr->Put("TotalTime", std::to_string(time));
            infoPtr->WriteInfo();
            infoPtr->Clear();
#endif

            predictions.push_back(time);
        }

        return predictions;

    }


    /* BROADCAST */

    //TODO: Consider nnz estimator class + template to make switching between things here easier
    template <typename IT, typename NT, typename DER>
    float BcastTime(CommModel<IT> * bcastModel, SpParMatInfoAnalytical<IT,NT,DER>& Minfo, SpGEMMParams& params, bool row) {

#ifdef PROFILE
        if (row)
            infoPtr->StartTimer("bcastCalcTime-A");
        else
            infoPtr->StartTimer("bcastCalcTime-B");
#endif

        std::vector<IT> * nnz2D = Minfo.GetNnzArr();

        // Compute local bcast times
        std::vector<float> locBcastTimes(params.GetTotalProcs());
        for (int p=0; p<params.GetTotalProcs(); p++) {
            
            // Vector containing nnz for each rank participating in broadcasts with rank p
            std::vector<IT> nnzBcastWorld(params.GetGridDim());
            //TODO: Params class should have methods that return ranks in row/col, then just use std::transform to create bcast world
            if (row) 
                nnzBcastWorld = Minfo.SliceNnzRow(nnz2D, p, params.GetGridDim());
            else
                nnzBcastWorld = Minfo.SliceNnzCol(nnz2D, p, params.GetGridDim());
            
            // Compute and sum all times for all bcasts rank p participates in 
            float locBcastTime = std::reduce(nnzBcastWorld.begin(), nnzBcastWorld.end(), 0, 
                [&Minfo, &bcastModel, &params](float sum, IT nnz) {
                    IT msgSize = Minfo.ComputeMsgSize(nnz);

                    CommOpts * opts = new CommOpts{
                        //gridSize <= params.GetCoresPerNode() ? true : false //intranode
                        false
                    };

                    CommInfo<IT> * info = MakeBcastCommInfo(params.GetGridDim(),  msgSize); 

                    float singleBcastTime = bcastModel->Time(info, opts);

                    delete info;
                    delete opts;

                    return singleBcastTime + sum;
                }
            );
            
            locBcastTimes[p] = locBcastTime;

        }

        // Reduce to get max time
        float finalTime = std::reduce(locBcastTimes.begin(), locBcastTimes.end(), 0,
            [](float currMax, float currElem) {
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
    float LocalMultTime(LocalSpGEMMModel<AIT, BIT>* model, 
                            SpParMatInfoAnalytical<AIT,ANT,ADER>& Ainfo,
                            SpParMatInfoAnalytical<BIT,BNT,BDER>& Binfo,
                            SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("multCalcTime");
#endif
        
        auto Adims = Ainfo.GetGridDims(); 
        auto Bdims = Binfo.GetGridDims();

        const int totalProcs = params.GetTotalProcs();

        std::vector<float> * localSpGEMMTimes = new std::vector<float>;
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
                                                    Ainfo.ComputeLocNnzGrid(NNZ_ARR,rankA), 
                                                    Binfo.ComputeLocNnzGrid(NNZ_ARR,rankB),
                                                    Ainfo.GetGlobDensity(),
                                                    Ainfo.GetLocDensityArr()->at(rankA),
                                                    Binfo.GetGlobDensity(),
                                                    Binfo.GetLocDensityArr()->at(rankB)};
                info->SetFLOPS(params, FLOPS_LOC_DENSITY);
                localSpGEMMTimes->push_back(model->Time(info));
            }

        }


        // Reduce to get max time
        float finalTime = std::reduce(localSpGEMMTimes->begin(),localSpGEMMTimes->end(), 0,
            [](float currMax, float currElem) {
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

    float LayerMergeTime() {
        return 0;
    }
 
};


#ifdef XGB_MODEL

class SpGEMM2DModelXgb : public SpGEMM2DModel<SpGEMM2DModelXgb> {
public:

    void CreateImpl() {

        XGB_CHECK(XGBoosterCreate(nullptr, 0, &bstHandle));

        //TODO: Get rid of hardcoded filepath
        const char * modelPath = "../include/CombBLAS/Autotuning/model/model_2d_xgb_globals.model";
        XGB_CHECK(XGBoosterLoadModel(bstHandle, modelPath));
        
        XGB_CHECK(XGBoosterGetNumFeature(bstHandle, (bst_ulong*)(&nFeatures)));
#ifdef DEBUG
        debugPtr->Print0("Num features: " + std::to_string(nFeatures));
#endif
    }

    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoXgb : public SpParMatInfo<IT,NT,DER> {
    public:
        
        using SpParMatInfo<IT,NT,DER>::nnz;
        using SpParMatInfo<IT,NT,DER>::ncols;
        using SpParMatInfo<IT,NT,DER>::nrows;
        using SpParMatInfo<IT,NT,DER>::globDensity;

        SpParMatInfoXgb(SpParMat<IT,NT,DER>& Mat):
            SpParMatInfo<IT,NT,DER>(Mat)
        {
            
            featureMap.emplace("nnz", nnz);
            featureMap.emplace("m", nrows);
            featureMap.emplace("n", ncols);
            featureMap.emplace("density", globDensity);

            SetGlobalColInfo(Mat);    
        }


        // NOTE: need overloaded function here because behavior differs depending on 2d vs 3d
        void SetGlobalColInfo(SpParMat<IT,NT,DER>& Mat) {
#ifdef PROFILE
            infoPtr->StartTimer("FeatureCollection");
#endif

            // avg nnz per column
            avgNnzCol = static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getncol());

            featureMap.emplace("avgNnzCol", avgNnzCol);

            // avg density per column
            avgDensityCol = (static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getnrow())) / 
                                    static_cast<float>(Mat.getncol());

            featureMap.emplace("avgDensityCol", avgDensityCol);

            // Reduce to get complete nnz per column
            std::vector<IT> nnzColVec(Mat.seqptr()->getncol());
            float sumNnzMeanDiff;

            for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
                nnzColVec[colIter.colid()] = colIter.nnz();
                sumNnzMeanDiff += std::pow( (colIter.nnz() - avgNnzCol), 2);
            }

            MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzColVec.data()), nnzColVec.size(), MPIType<IT>(), MPI_SUM,
                        Mat.getcommgrid()->GetColWorld());

            // Compute column densities
            std::vector<float> densityColVec(Mat.seqptr()->getncol());
            float sumDensityMeanDiff;

            std::transform(nnzColVec.begin(), nnzColVec.end(), densityColVec.begin(),
                    [this, &sumDensityMeanDiff](IT nnz) mutable {
                        float d = static_cast<float>(nnz) / static_cast<float>(this->nrows);
                        sumDensityMeanDiff += std::pow( (d - this->avgDensityCol), 2);
                        return d;
                    }
            );

            // Local reduce to get min, max and sum for each column block
            float locMinDensity, locMaxDensity;
            minNnzCol = ReduceMin(nnzColVec);
            maxNnzCol = ReduceMax(nnzColVec);
            minDensityCol = ReduceMin(densityColVec);
            maxDensityCol = ReduceMax(densityColVec);

            // Global reduce to compute final min, max, and sum
            // TODO: use nonblocking collectives?
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&minNnzCol), 1, MPIType<IT>(), MPI_MIN, 
                            Mat.getcommgrid()->GetRowWorld());
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxNnzCol), 1, MPIType<IT>(), MPI_MAX, 
                            Mat.getcommgrid()->GetRowWorld());

            MPI_Allreduce(MPI_IN_PLACE, (void*)(&minDensityCol), 1, MPI_FLOAT, MPI_MIN, 
                            Mat.getcommgrid()->GetRowWorld());
            MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxDensityCol), 1, MPI_FLOAT, MPI_MAX, 
                            Mat.getcommgrid()->GetRowWorld());

            // pack floats that will be summed into single buffer
            float locBuf[] = {sumNnzMeanDiff, sumDensityMeanDiff};
            MPI_Allreduce(MPI_IN_PLACE, (void*)(locBuf), 2, MPI_FLOAT, MPI_SUM, Mat.getcommgrid()->GetRowWorld());

            // finish stdev calculations
            stdevNnzCol = std::sqrt( sumNnzMeanDiff / Mat.getncol() );
            stdevDensityCol = std::sqrt( sumDensityMeanDiff / Mat.getncol() );
            
            featureMap.emplace("minNnzCol", minNnzCol);
            featureMap.emplace("maxNnzCol", maxNnzCol);
            featureMap.emplace("minDensityCol", minDensityCol);
            featureMap.emplace("maxDensityCol", maxDensityCol);
            featureMap.emplace("stdevNnzCol", stdevNnzCol);
            featureMap.emplace("stdevDensityCol", stdevDensityCol);

#ifdef PROFILE
            infoPtr->EndTimer("FeatureCollection");
            infoPtr->Print("FeatureCollection");
#endif

        }


        inline float GetAvgNnzCol() const {return avgNnzCol;}
        inline IT GetMinNnzCol() const {return minNnzCol;}
        inline IT GetMaxNnzCol() const {return maxNnzCol;}
        inline float GetStdevNnzCol() const {return stdevNnzCol;}

        inline float GetAvgDensityCol() const {return avgDensityCol;}
        inline float GetMinDensityCol() const {return minDensityCol;}
        inline float GetMaxDensityCol() const {return maxDensityCol;}
        inline float GetStdevDensityCol() const {return stdevDensityCol;}

        inline std::map<std::string, float> GetFeatureMap() const {return featureMap;} 

    private:

        float avgNnzCol;
        IT minNnzCol;
        IT maxNnzCol;
        float stdevNnzCol;
        float avgDensityCol;
        float minDensityCol;
        float maxDensityCol;
        float stdevDensityCol;

        std::map<std::string, float> featureMap;

    };

    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {
    public:
        Inputs(SpParMat<AIT,ANT,ADER>& A, SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A),Binfo(B)
        {
        }

        SpParMatInfoXgb<AIT,ANT,ADER> Ainfo;
        SpParMatInfoXgb<BIT,BNT,BDER> Binfo;
        
    };

    std::vector<float> PredictImpl(std::vector<float>& X) {

        // Create DMat
        int nSamples = X.size() / nFeatures;
        DMatrixHandle dMatHandle;
        XGB_CHECK(XGDMatrixCreateFromMat(X.data(), nSamples, nFeatures, 0.0, &dMatHandle)); 

        // Make prediction
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";
        bst_ulong outDim;
        const bst_ulong * outShape; 
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(bstHandle, dMatHandle, config, &outShape, &outDim, &prediction));

        return std::vector<float>(prediction, prediction+nSamples);

    }


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> MakeFeatureMatImpl(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, 
                                            std::vector<SpGEMMParams>& searchSpace) {

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;
        
        int nSamples = searchSpace.size();

        // Feature order
        std::vector<std::string> featureOrder{
            "avgDensityCol",
            "avgNnzCol",
            "density",
            "m",
            "maxDensityCol",
            "maxNnzCol",
            "minDensityCol",
            "minNnzCol",
            "n",
            "nnz",
            "stdevDensityCol",
            "stdevNnzCol"
        };

        // Each row is a sample
        std::vector<float> featureMat;
        featureMat.reserve(nSamples*nFeatures);

        //TODO: There has to be a better way to do this
        // Populate the feature matrix
        for (int i=0; i<nSamples; i++) {

            // Nodes and PPN always go first
            auto currParams = searchSpace[i];
            featureMat.push_back(currParams.GetNodes());
            featureMat.push_back(currParams.GetPPN());
            
            // Iterate through features in this sample according to feature order defined earlier
            // and push them onto the matrix
            std::for_each(featureOrder.begin(), featureOrder.end(),
                [&featureMat, &Ainfo, &Binfo](auto& featureName) {
                    // Order is always feature-A, feature-B
                    featureMat.push_back(Ainfo.GetFeatureMap()[featureName]);
                    featureMat.push_back(Binfo.GetFeatureMap()[featureName]);
                }
            );
        }


        return featureMat; 
    }


private:
    int nFeatures;
    BoosterHandle bstHandle;

};


class SpGEMM2DModelPhase : public SpGEMM2DModel<SpGEMM2DModelPhase> {

public:

    void CreateImpl() {
        XGB_CHECK(XGBoosterCreate(nullptr, 0, &multBstHandle));
        XGB_CHECK(XGBoosterCreate(nullptr, 0, &mergeBstHandle));

        //TODO: Remove hardocded filepaths
        const char * multModelPath = "../include/CombBLAS/Autotuning/model/models/xgb-mult.model";
        const char * mergeModelPath = "../include/CombBLAS/Autotuning/model/models/xgb-merge.model";
        
        XGB_CHECK(XGBoosterLoadModel(multBstHandle, multModelPath));
        XGB_CHECK(XGBoosterLoadModel(mergeBstHandle, mergeModelPath));

        XGB_CHECK(XGBoosterGetNumFeature(multBstHandle, (bst_ulong*)(&nFeatures)));

        std::vector<std::string> features{
            "FLOPS",
            "m-A",
            "m-B",
            "n-A",
            "n-B",
            "nnz-A",
            "nnz-B",
            "outputNnz-intermediate",
            "outputNnz-final",
            "Nodes",
            "PPN",
        };

        ASSERT(nFeatures==features.size(), "Feature size is wrong");
    }
    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoPhase : public SpParMatInfo<IT,NT,DER> {
    public:
        
        using SpParMatInfo<IT,NT,DER>::locNnz;
        using SpParMatInfo<IT,NT,DER>::locNrowsExact;
        using SpParMatInfo<IT,NT,DER>::locNcolsExact;
        using SpParMatInfo<IT,NT,DER>::rank;
        using SpParMatInfo<IT,NT,DER>::colRank;
        using SpParMatInfo<IT,NT,DER>::rowRank;
        using SpParMatInfo<IT,NT,DER>::ncols;
        using SpParMatInfo<IT,NT,DER>::nrows;

        SpParMatInfoPhase(SpParMat<IT,NT,DER>& Mat):
            SpParMatInfo<IT,NT,DER>(Mat)
        {
            gridComm = Mat.getcommgrid()->GetWorld();
            worldSize = Mat.getcommgrid()->GetSize();
        }

        MPI_Comm gridComm;
        int worldSize;

    };


    //TODO: One day, this will all need a semiring template parameter
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {
    public:
        
        typedef PlusTimesSRing<ANT,BNT> PTTF;

        Inputs(SpParMat<AIT,ANT,ADER>& A, SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A), Binfo(B), FLOPS(0), outputNnzIntermediate(0), outputNnzFinal(0)
        {

#ifdef PROFILE
            infoPtr->StartTimerGlobal("FLOPEstimation");
#endif

            EstimateFLOP<PTTF, AIT, ANT, BNT, ADER, BDER>(A,B,false,false,&FLOPS);

#ifdef PROFILE
            infoPtr->EndTimerGlobal("FLOPEstimation");
#endif

#ifdef PROFILE
            infoPtr->StartTimerGlobal("NnzIntermediate");
#endif
			AIT * flopC = estimateFLOP(*(A.seqptr()), *(B.seqptr())); 
 
			if (!(A.seqptr()->isZero()) && !(B.seqptr()->isZero())) {
                AIT * outputNnzCol = estimateNNZ_Hash(*(A.seqptr()), *(B.seqptr()), flopC);
				for (int i=0; i<B.seqptr()->GetDCSC()->nzc; i++)
				{
					outputNnzIntermediate += outputNnzCol[i];
				}
			}

            
#ifdef PROFILE
            infoPtr->EndTimerGlobal("NnzIntermediate");
#endif

#ifdef PROFILE
            infoPtr->StartTimerGlobal("NnzFinal");
#endif

            outputNnzFinal = EstPerProcessNnzSUMMAMax(A,B,false);

#ifdef PROFILE
            infoPtr->EndTimerGlobal("NnzFinal");
#endif
        
        }

        SpParMatInfoPhase<AIT,ANT,ADER> Ainfo;
        SpParMatInfoPhase<BIT,BNT,BDER> Binfo;
        
        AIT FLOPS;
        AIT outputNnzIntermediate;
        AIT outputNnzFinal;

    };


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> PredictImpl(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs,
                                    std::vector<SpGEMMParams>& searchSpace) {

        std::vector<float> times(searchSpace.size());
#ifdef TIMING
        infoPtr->StartTimerGlobal("Prediction");
#endif
        std::transform(searchSpace.begin(), searchSpace.end(), times.begin(),
            [&inputs, this](auto& params) {

                auto featureMat = this->MakeFeatureMatImpl(inputs, params);

                DMatrixHandle featureMatHandle;
                XGB_CHECK(XGDMatrixCreateFromMat(featureMat.data(), params.GetTotalProcs(), nFeatures, 0.0,
                                                    &featureMatHandle));
                
                // Each of these do a prediction for the entire grid
                auto bcastTimes = this->BcastTime<AIT, ANT>(featureMat, params); // Don't need Dmat here
                auto localSpGEMMTimes = this->LocalSpGEMMTime(featureMatHandle, params);
                auto mergeTimes = this->MergeTime(featureMatHandle, params);
                
                // Sum all times, then return the max
                std::vector<float> paramTimes(params.GetTotalProcs());
                std::transform(bcastTimes.begin(), bcastTimes.end(), localSpGEMMTimes.begin(),
                                paramTimes.begin(), std::plus<>());
                std::transform(paramTimes.begin(), paramTimes.end(), mergeTimes.begin(), paramTimes.begin(),
                                std::plus<>());
#ifdef TIMING
                infoPtr->WriteInfo();
                infoPtr->Clear();
#endif

                return ReduceMax(paramTimes);
            }
        );

#ifdef TIMING
        infoPtr->EndTimerGlobal("Prediction");
#endif
        return times;

    }


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    std::vector<float> MakeFeatureMatImpl(Inputs<AIT,ANT,ADER, BIT,BNT,BDER>& inputs,
                                            SpGEMMParams& params) {

#ifdef DEBUG
        debugPtr->Print0(params.OutStr());
#endif

#ifdef PROFILE
        infoPtr->StartTimer("FeatureCollection");
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        std::vector<float> featureMat(nFeatures*params.GetTotalProcs());

        // For now, assume always scaling down
        ASSERT(jobPtr->totalTasks>=params.GetTotalProcs(), "Scaling up is not yet supported");

        int gridDim = params.GetGridDim();
        int superTileDim = RoundedSqrt<int,int>(Ainfo.worldSize) / gridDim;

        auto SuperTileColor = [&gridDim, &superTileDim](int rowRank, int colRank) {
            return ( (rowRank / superTileDim) ) + ( ((colRank) / superTileDim) * gridDim );
        };

        auto SuperTileKey = [&gridDim, &superTileDim](int rowRank, int colRank) {
            return ( ((rowRank ) % superTileDim) + ((colRank % superTileDim) * superTileDim ) );
        };

        // Make communicators corresponding to each supertile
        MPI_Comm superTileComm;
        MPI_Comm_split(Ainfo.gridComm, 
                        SuperTileColor(Ainfo.rowRank, Ainfo.colRank),
                        SuperTileKey(Ainfo.rowRank, Ainfo.colRank),
                        &superTileComm);
                        
        // Pack everything into a single buffer
        int msgCount = 9;
        float sendBuf[] = {(const float)inputs.FLOPS, 
                            (const float)Ainfo.locNrowsExact,
                            (const float)Binfo.locNrowsExact,
                            (const float)Ainfo.locNcolsExact,
                            (const float)Binfo.locNcolsExact,
                            (const float)Ainfo.locNnz, 
                            (const float)Binfo.locNnz, 
                            (const float)inputs.outputNnzIntermediate, 
                            (const float)inputs.outputNnzFinal};
        float * recvBuf = new float[msgCount];

        // Reduce into top left corner of each supertile
        MPI_Reduce((void*)(sendBuf), (void*)(recvBuf), msgCount, MPI_FLOAT, MPI_SUM, 0, superTileComm);

        // Local sample to be gathered into featureMat
        float locSample[] = {recvBuf[0],
                             recvBuf[1]/(float)std::sqrt(Ainfo.worldSize),
                             recvBuf[2]/(float)std::sqrt(Ainfo.worldSize),
                             recvBuf[3]/(float)std::sqrt(Ainfo.worldSize),
                             recvBuf[4]/(float)std::sqrt(Ainfo.worldSize),
                             recvBuf[5],
                             recvBuf[6],
                             recvBuf[7],
                             recvBuf[8],
                             (const float)params.GetNodes(),
                             (const float)params.GetPPN()};

        //Communicator consisting of rank 0 and all top left corner ranks
        MPI_Group worldGroup;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
        
        std::vector<int> topLeftRanks(Ainfo.worldSize / (superTileDim*superTileDim));
        for (int i=0; i<topLeftRanks.size(); i++) {
            topLeftRanks[i] = ((i*superTileDim)%RoundedSqrt<int,int>(Ainfo.worldSize)) 
                            + ((i / gridDim) * (gridDim * superTileDim * superTileDim));
        }

        MPI_Group topLeftGroup;
        MPI_Group_incl(worldGroup, topLeftRanks.size(), topLeftRanks.data(), &topLeftGroup);
        
        MPI_Comm topLeftComm;
        MPI_Comm_create(MPI_COMM_WORLD, topLeftGroup, &topLeftComm);

        // Gather into featureMat on rank 0
        if (topLeftComm!=MPI_COMM_NULL) {
            MPI_Gather((void*)(locSample), nFeatures, MPI_FLOAT, (void*)(featureMat.data()), nFeatures,
                        MPI_FLOAT, 0, topLeftComm);
        }


        //TODO: REMOVE THIS BCAST, it should not be necessary
        MPI_Bcast((void*)(featureMat.data()), featureMat.size(), MPI_FLOAT, 0, Ainfo.gridComm);

#ifdef PROFILE
        infoPtr->EndTimer("FeatureCollection");
#endif
#ifdef DEBUG
        debugPtr->LogVecSameLine(featureMat, "FeatureMat");
#endif

        MPI_Barrier(Ainfo.gridComm);

        return featureMat;

    }

    template <typename IT, typename NT>
    std::vector<float> BcastTime(std::vector<float>& X, SpGEMMParams& params) {
        
#ifdef PROFILE
        infoPtr->StartTimer("BcastCompute");
#endif

        auto TreeBcast = [this](int commSize, IT msgSize) {
            float alpha = this->platformParams.GetInternodeAlpha() * std::log2(commSize);
            float beta = (std::log2(commSize) * msgSize) / this->platformParams.GetInternodeBeta();
            return (alpha + beta);
        };

        auto MsgSize = [](IT nnz) {
            return nnz*sizeof(NT) + nnz*sizeof(IT) + (nnz + 1) * sizeof(IT);
        };


        // Compute each local bcast time
        std::vector<float> timesA(params.GetGridDim());
        std::vector<float> timesB(params.GetGridDim());
        for (int k=0; k<params.GetTotalProcs(); k++) {

            IT nnzA = static_cast<IT>(X[k*nFeatures + 5]); //TODO: Hardcoding these numbers makes me ill
            IT nnzB = static_cast<IT>(X[k*nFeatures + 6]);

            IT bytesA = MsgSize(nnzA);
            IT bytesB = MsgSize(nnzB);
            
            float bcastTimeA = TreeBcast(params.GetGridDim(), bytesA); 
            float bcastTimeB = TreeBcast(params.GetGridDim(), bytesB); 
            
            int i = k % params.GetGridDim();
            int j = k / params.GetGridDim();

            timesA[i] += bcastTimeA;
            timesB[j] += bcastTimeB;

        }


        // Compute final array of bcast times
        std::vector<float> finalTimes(params.GetTotalProcs());
        for (int k=0; k<finalTimes.size(); k++) {
            
            int i = k % params.GetGridDim();
            int j = k / params.GetGridDim();

            finalTimes[k] = timesA[i] + timesB[j];

        }
        
#ifdef PROFILE
        infoPtr->EndTimer("BcastCompute");
#endif

        return finalTimes;
    }

    
    std::vector<float> LocalSpGEMMTime(DMatrixHandle& X, SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("MultCompute");
#endif

        //TODO: Does this matter?
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";

        bst_ulong outDim;
        const bst_ulong * outShape;
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(multBstHandle, X, config, &outShape, &outDim, &prediction));

#ifdef PROFILE
        infoPtr->EndTimer("MultCompute");
#endif

        return std::vector<float>(prediction, prediction+params.GetTotalProcs());
  
    }


    std::vector<float> MergeTime(DMatrixHandle& X, SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("MergeCompute");
#endif
        //TODO: Does this matter?
        char const config[] =
        "{\"training\": false, \"type\": 0, "
        "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";

        bst_ulong outDim;
        const bst_ulong * outShape;
        const float * prediction;
        XGB_CHECK(XGBoosterPredictFromDMatrix(mergeBstHandle, X, config, &outShape, &outDim, &prediction));

#ifdef PROFILE
        infoPtr->EndTimer("MergeCompute");
#endif

        return std::vector<float>(prediction, prediction+params.GetTotalProcs());
    }


private:
    int nFeatures; // same number of features for both models
    BoosterHandle multBstHandle;
    BoosterHandle mergeBstHandle;

};

#endif


}//autotuning
}//combblas

#endif





