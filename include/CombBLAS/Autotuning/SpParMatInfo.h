
#ifndef SPPARMATINFO_H
#define SPPARMATINFO_H

#include "common.h"
#include "SpGEMMParams.h"


#define NNZ_THRESH 0
#define NNZ_TUPLES_COL
//#define NNZ_TUPLES_ROW

namespace combblas {
namespace autotuning {

enum NNZ_STRAT {NNZ_GLOB_DENSITY, NNZ_LOC_DENSITY, NNZ_ARR};

template <typename IT, typename NT, typename DER>
class SpParMatInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;

    /* (row,col,nnz) */  
    //TODO: For col split, no need to store row idx, and for row split, no need to store col idx
    typedef std::vector<std::tuple<IT,IT,IT>> NnzTuples;

    enum SPLIT {COL_SPLIT, ROW_SPLIT}; 


    //NOTE: loc* are values for the actual 2D processor grid
    //distInfo determines if distribution-specific information, like the array of tile densities, is computed
    SpParMatInfo(SpParMat3D<IT,NT,DER>& Mat, bool distInfo=true):
        
        locMat(Mat.seqptr()),

        locNnz(Mat.seqptr()->getnnz()), 

        locNcols(Mat.getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat.getnrow() / RoundedSqrt<IT,IT>(worldSize)),
        locNcolsExact(Mat.seqptr()->getncol()),
        locNrowsExact(Mat.seqptr()->getnrow()),

        nnzArr(new std::vector<IT>(0)),
        locDensityArr(new std::vector<float>(worldSize)),

        rowRank(Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcRow()),
        colRank(Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcCol())

    {

        SetGlobalInfo(Mat);

        SetGlobalColInfo(Mat);
        
        if (distInfo) { 

            locDensityArr->insert(locDensityArr->begin() + rank, 
                                    static_cast<float>(locNnz) / static_cast<float>(locNcolsExact*locNrowsExact));
            MPI_Allgather(MPI_IN_PLACE, 1, MPI_FLOAT, (void*)(locDensityArr->data()), 1, MPI_FLOAT, MPI_COMM_WORLD); 

            split = Mat.isColSplit() ? COL_SPLIT : ROW_SPLIT;
            
            if (split==COL_SPLIT) {

#ifdef NNZ_TUPLES_COL

#ifdef PROFILE
                infoPtr->StartTimer("nnzTuplesColInit");
#endif
                nnzTuples = NnzTuplesCol(); 
#ifdef PROFILE
                infoPtr->EndTimer("nnzTuplesColInit");
#endif

#endif

            } else if (split==ROW_SPLIT) {

#ifdef NNZ_TUPLES_ROW

#ifdef PROFILE
                infoPtr->StartTimer("nnzTuplesRowInit");
#endif
                nnzTuples = NnzTuplesRow();
#ifdef PROFILE
                infoPtr->EndTimer("nnzTuplesRowInit");
#endif

#endif
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

    }


    SpParMatInfo(SpParMat<IT,NT,DER>& Mat, bool distInfo=true):

        locMat(Mat.seqptr()),
        locNnz(Mat.seqptr()->getnnz()), 
        locNcols(Mat.getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat.getnrow() / RoundedSqrt<IT,IT>(worldSize)),
        locNcolsExact(Mat.seqptr()->getncol()),
        locNrowsExact(Mat.seqptr()->getnrow()),

        nnzArr(new std::vector<IT>(0)),
        locDensityArr(new std::vector<float>(worldSize)),

        rowRank(Mat.getcommgrid()->GetRankInProcRow()),
        colRank(Mat.getcommgrid()->GetRankInProcCol())
    {

        SetGlobalInfo(Mat);

        SetGlobalColInfo(Mat);
        
        if (distInfo) {

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

    }

    
    template <typename M>
    void SetGlobalInfo(M& Mat) {
        this->nnz = Mat.getnnz();
        this->ncols = Mat.getncol();
        this->nrows = Mat.getnrow();
        this->globDensity = static_cast<float>(this->nnz) / static_cast<float>(this->ncols*this->nrows);
    }

    
    // NOTE: need overloaded function here because behavior differs depending on 2d vs 3d
    void SetGlobalColInfo(SpParMat<IT,NT,DER>& Mat) {

        // avg nnz per column 
        avgNnzCol = static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getncol());

        // avg density per column
        avgDensityCol = (static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getnrow())) / static_cast<float>(Mat.getncol());
        
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
        MPI_Allreduce(MPI_IN_PLACE, (void*)(&minNnzCol), 1, MPIType<IT>(), MPI_MIN, Mat.getcommgrid()->GetRowWorld());
        MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxNnzCol), 1, MPIType<IT>(), MPI_MAX, Mat.getcommgrid()->GetRowWorld());

        MPI_Allreduce(MPI_IN_PLACE, (void*)(&minDensityCol), 1, MPI_FLOAT, MPI_MIN, Mat.getcommgrid()->GetRowWorld());
        MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxDensityCol), 1, MPI_FLOAT, MPI_MAX, Mat.getcommgrid()->GetRowWorld());

        // pack floats that will be summed into single buffer
        float locBuf[] = {sumNnzMeanDiff, sumDensityMeanDiff};
        MPI_Allreduce(MPI_IN_PLACE, (void*)(locBuf), 2, MPI_FLOAT, MPI_SUM, Mat.getcommgrid()->GetRowWorld());
        
        // finish stdev calculations
        stdevNnzCol = std::sqrt( sumNnzMeanDiff / Mat.getncol() );
        stdevDensityCol = std::sqrt( sumDensityMeanDiff / Mat.getncol() );

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


    //TODO: Pass SpGEMMParams instead of ppn nodes and layers so we don't have to constantly recompute stuff
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


    IT ComputeMsgSize(const int locNnz) {
        return locNnz * GetNzvalSize() +
                locNnz * GetIndexSize() +
                (locNnz + 1) * GetIndexSize();
    }


    //TODO: This really needs to be partitioned better. Logic for "computing stuff given 3d grid" is inconsistent
    //TODO: All these member functions should accept SpGEMMParams instance as argument instead of ppn, nodes, layers..
    // row, column
    void SetGridDims(SpGEMMParams& params) {

        /* Info about currently, actually formed 2D processor grid */
        const int totalProcs2D = jobPtr->totalTasks;
        const int procCols2D = RoundedSqrt<int,int>(totalProcs2D);
        const int procRows2D = procCols2D;

        /* Info about 3D grid */
        const int ppn = params.GetPPN();
        const int nodes = params.GetNodes();
        const int layers = params.GetLayers();
        const int totalProcs = params.GetTotalProcs();
        const int gridSize = params.GetGridSize();
        const int gridRows = params.GetGridDim();
        const int gridCols = gridRows;

        IT locNcolsGrid;
        IT locNrowsGrid;

        if (split==COL_SPLIT) {
            locNcolsGrid = locNcols * (procCols2D/gridCols)/layers;
            locNrowsGrid = locNrows * (procRows2D/gridRows);
        } else if (split==ROW_SPLIT) {
            locNcolsGrid = locNcols * (procCols2D/gridCols);
            locNrowsGrid = locNrows * (procRows2D/gridRows)/layers;
        } else {
            UNREACH_ERR();
        }

        gridDims = std::make_pair(locNrowsGrid, locNcolsGrid);

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


    /* Vector of ranks in processor row procRank belongs to, including procRank */
    std::vector<int> RowRanks(const int procRank, const SpGEMMParams& params) {

        std::vector<int> ranks;
        ranks.reserve(params.GetGridDim()); 
        
        for (int p=0; p<params.GetGridDim(); p++) {
            int currRank = procRank / params.GetGridDim() + p;
            ranks.push_back(currRank);
        }
        
        return ranks;
        
    }


    /* Vector of ranks in processor column procRank belongs to, including procRank */
    std::vector<int> ColRanks(const int procRank, const SpGEMMParams& params) {

        std::vector<int> ranks;
        ranks.reserve(params.GetGridDim()); 
        
        for (int p=0; p<params.GetGridDim(); p++) {
            int currRank = procRank % params.GetGridDim() + (p*params.GetGridDim());
            ranks.push_back(currRank);
        }
        
        return ranks;
        
    }


    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}

    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}
    inline float GetGlobDensity() const {return globDensity;}

    inline IT GetAvgNnzCol() const {return avgNnzCol;}
    inline IT GetMinNnzCol() const {return minNnzCol;}
    inline IT GetMaxNnzCol() const {return maxNnzCol;}
    inline IT GetStdevNnzCol() const {return stdevNnzCol;}

    inline IT GetAvgDensityCol() const {return avgDensityCol;}
    inline IT GetMinDensityCol() const {return minDensityCol;}
    inline IT GetMaxDensityCol() const {return maxDensityCol;}
    inline IT GetStdevDensityCol() const {return stdevDensityCol;}

    inline IT GetLocNnz() const {return locNnz;}
    inline IT GetLocNcols() const {return locNcols;}
    inline IT GetLocNrows() const {return locNrows;}

    inline std::vector<float> * GetLocDensityArr() const {return locDensityArr;}

    inline SPLIT GetSplit() const {return split;}

    inline std::vector<IT> * GetNnzArr() {return nnzArr;}

    inline std::pair<IT,IT> GetGridDims() {return gridDims;}

private:

    // Global info
    IT nnz;
    IT ncols;
    IT nrows;
    float globDensity;

    // Global column info
    float avgNnzCol;
    IT minNnzCol;
    IT maxNnzCol;
    float stdevNnzCol;
    float avgDensityCol;
    float minDensityCol;
    float maxDensityCol;
    float stdevDensityCol;
    
    // Info about actual 2D grid
    IT locNnz;
    IT locNcols;
    IT locNrows;
    IT locNcolsExact;
    IT locNrowsExact;
    int rowRank; //rank in actual 2d grid
    int colRank; //^^
    std::vector<float> * locDensityArr;
    DER * locMat;
    NnzTuples * nnzTuples;

    // Row or column split
    SPLIT split;    

    // Stores nnz per processor in hypothetical 3D grid
    std::vector<IT> * nnzArr;

    // Dimensions of tile in hypothetical 3D grid
    std::pair<IT,IT> gridDims;
    
};


}//autotuning
}//combblas


#endif
