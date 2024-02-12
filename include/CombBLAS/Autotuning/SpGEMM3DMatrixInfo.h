
#ifndef SPGEMM3DMATRIXINFO_H
#define SPGEMM3DMATRIXINFO_H

#include "common.h"
#include "SpGEMM3DParams.h"


#define NNZ_THRESH 0
#define NNZ_MAT_COL
//#define NNZ_MAT_ROW

namespace combblas {
namespace autotuning {

enum NNZ_STRAT {NNZ_GLOB_DENSITY, NNZ_LOC_DENSITY, NNZ_ARR};

template <typename IT, typename NT, typename DER>
class SpGEMM3DMatrixInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;

    /* (row,col,nnz) */  
    //TODO: For col split, no need to store row idx, and for row split, no need to store col idx
    typedef std::vector<std::tuple<IT,IT,IT>> NnzTuples;

    enum SPLIT {COL_SPLIT, ROW_SPLIT}; 


    //NOTE: loc* are values for the actual 2D processor grid
    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& Mat):
        
        nnz(Mat.getnnz()), ncols(Mat.getncol()), nrows(Mat.getnrow()),

        locNnz(Mat.seqptr()->getnnz()), 

        locNcols(Mat.getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat.getnrow() / RoundedSqrt<IT,IT>(worldSize)),

        locNcolsExact(Mat.seqptr()->getncol()),
        locNrowsExact(Mat.seqptr()->getnrow()),

        locMat(Mat.seqptr()),

        rowRank(Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcRow()),
        colRank(Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcCol()),

        nnzArr(new std::vector<IT>(0))
     {
        
        INIT_TIMER();

        globDensity = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        locDensity = static_cast<float>(locNnz) / static_cast<float>(locNcolsExact*locNrowsExact);
        split = Mat.isColSplit() ? COL_SPLIT : ROW_SPLIT;
        
        if (split==COL_SPLIT) {
#ifdef NNZ_MAT_COL
            START_TIMER();
            nnzTuples = NnzTuplesCol(); 
            END_TIMER("nnzTuplesCol Init Time: ");
#endif
        } else if (split==ROW_SPLIT) {
#ifdef NNZ_MAT_ROW
            START_TIMER();
            nnzTuples = NnzTuplesRow();
            END_TIMER("nnzTuplesRow Init Time: ");
#endif
        }

        MPI_Barrier(MPI_COMM_WORLD);

    }


    /* Create sparse matrix storing nnz for each block row of each column and distribute across all ranks  */
    NnzTuples * NnzTuplesCol() {

        INIT_TIMER();

        START_TIMER();

        auto _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
        _nnzTuples->reserve(locNcolsExact);
        
        // Init local data
        int locTupleSize = 0;
        for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
            if (colIter.nnz()>NNZ_THRESH) {
                _nnzTuples->push_back( std::tuple<IT,IT,IT>{colRank,  colIter.colid() + locNcols*rowRank, colIter.nnz()} ); 
            }
        }

        END_TIMER("Time for local tuple construction in col mat: ");

#ifdef DEBUG
        debugPtr->Log("locNnzTuples col");
        for (int i=0; i<_nnzTuples->size(); i++) {
            debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
        }
#endif

        return _nnzTuples;

    }


    /* Initialize array containing nnz per row on each processor, then gather on processor 0 */
    NnzTuples * NnzTuplesRow() {

        INIT_TIMER();

        START_TIMER();

        // JB: I can't figure out a way to avoid mutating nnz during iteration, so we can't just use std::tuple
        std::map<std::tuple<IT,IT>, IT> nnzMap;
        for (auto colIter = locMat->begcol(); colIter != locMat->endcol(); colIter++) {
            for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                std::tuple<IT,IT> t{nzIter.rowid() + locNrows*colRank, rowRank};
                nnzMap.emplace(t, 0);
                nnzMap[t] += 1;
            }
        }

        END_TIMER("Time for local nnzMap construction in row mat: ");

        START_TIMER();

        auto  _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
        _nnzTuples->reserve(locNrowsExact);

        std::for_each(nnzMap.begin(), nnzMap.end(), 
            [&_nnzTuples](auto& elem)  {
                std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
                _nnzTuples->push_back( t );
            }
        );
        
        END_TIMER("Time for local tuple construction in row mat: ");

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
    IT SetLocNnzGlobDensity() {

        IT localNcols = std::get<1>(dims3D);
        IT localNrows = std::get<0>(dims3D);
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(globDensity * localMatSize);
        return localNnzApprox;
    }


    /* Approximate local nnz using matrix globDensity
     * This actually just computes the avg nnz per processor
     */
    IT SetLocNnzLocDensity() {

        IT localNcols = std::get<1>(dims3D);
        IT localNrows = std::get<0>(dims3D);
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(locDensity * localMatSize);
        return localNnzApprox;
    }


    void SetNnzArr(SpGEMM3DParams& params) {

        INIT_TIMER();

        START_TIMER();
        
        nnzArr->clear();
        nnzArr->resize(params.GetTotalProcs());

        switch(split) {
            case COL_SPLIT:
            {
                SetNnzArrColSplit(params);
                break;
            }
            case ROW_SPLIT:
            {
                SetNnzArrRowSplit(params);
                break;
            }
            default:
            {
                UNREACH_ERR();
            }
        }

        END_TIMER("Compute 3D nnz time: ");
    }


    /* Given local nnz in initial 2D processor grid, compute nnz per processor in 3D processr grid
     * WITHOUT explicitly forming the 3D processor grid. */
    void SetNnzArrColSplit(SpGEMM3DParams& params) {

        const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_MAT_COL
        // Local nnz array
        std::for_each(nnzTuples->begin(), nnzTuples->end(), 
            [&params,this](auto& t) {
                int i = std::get<0>(t);
                int j = std::get<1>(t);
                int owner = ComputeOwner3D(params, i*this->locNrows, j, COL_SPLIT);
                this->nnzArr->at(owner) += std::get<2>(t);
            }
        );
#else
        // Just use local matrix
        for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
            int j = colIter.colid();
            for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                int i = nzIter.rowid();
                int owner = ComputeOwner3D(params, i+(colRank*locNrows), j+(rowRank*locNcols), COL_SPLIT);
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
        

    void SetNnzArrRowSplit(SpGEMM3DParams& params) {

        const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_MAT_ROW
        // Local data
        std::for_each(nnzTuples->begin(), nnzTuples->end(),
            [&params, this](auto& t) {
                int i = std::get<0>(t);
                int j = std::get<1>(t);
                int owner = ComputeOwner3D(params, i, j*this->locNcols, ROW_SPLIT);
                this->nnzArr->at(owner) += std::get<2>(t);
            }
        );
#else
        for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
            int j = colIter.colid();
            for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
                int i = nzIter.rowid();
                int owner = ComputeOwner3D(params, i+(colRank*locNrows), j+(rowRank*locNcols), ROW_SPLIT);
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

    
    IT GetLocNnz3D(NNZ_STRAT strat, int procRank) {
        switch(strat) {
            case NNZ_GLOB_DENSITY:
                return SetLocNnzGlobDensity();
            case NNZ_LOC_DENSITY:
                return SetLocNnzLocDensity();
            case NNZ_ARR:
                return nnzArr->at(procRank);
            default:
                UNREACH_ERR();
        }
        return 0;
    }


    //TODO: Pass SpGEMM3DParams instead of ppn nodes and layers so we don't have to constantly recompute stuff
    int ComputeOwner3D(SpGEMM3DParams& params, const int i, const int j, SPLIT split) {
        
        const int layers = params.GetLayers();
        const int gridDim = params.GetGridDim();
        const int gridSize = params.GetGridSize();

        IT locNrows3D = std::get<0>(dims3D);
        IT locNcols3D = std::get<1>(dims3D);

        IT colDiv;
        IT rowDiv;
        IT layerDiv;

        int layerIdx;
        
        if (split==COL_SPLIT) {
            colDiv = locNcols3D*layers;
            rowDiv = locNrows3D;
            layerDiv = locNcols3D;
            layerIdx = j;
        } else if (split==ROW_SPLIT) {
            colDiv = locNcols3D;
            rowDiv = locNrows3D*layers;
            layerDiv = locNrows3D;
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
    //TODO: All these member functions should accept SpGEMM3DParams instance as argument instead of ppn, nodes, layers..
    // row, column
    void SetDims3D(SpGEMM3DParams& params) {

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

        IT locNcols3D;
        IT locNrows3D;

        if (split==COL_SPLIT) {
            locNcols3D = locNcols * (procCols2D/gridCols)/layers;
            locNrows3D = locNrows * (procRows2D/gridRows);
        } else if (split==ROW_SPLIT) {
            locNcols3D = locNcols * (procCols2D/gridCols);
            locNrows3D = locNrows * (procRows2D/gridRows)/layers;
        } else {
            UNREACH_ERR();
        }

        dims3D = std::make_pair(locNrows3D, locNcols3D);

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
    std::vector<int> RowRanks(const int procRank, const SpGEMM3DParams& params) {

        std::vector<int> ranks;
        ranks.reserve(params.GetGridDim()); 
        
        for (int p=0; p<params.GetGridDim(); p++) {
            int currRank = procRank / params.GetGridDim() + p;
            ranks.push_back(currRank);
        }
        
        return ranks;
        
    }


    /* Vector of ranks in processor column procRank belongs to, including procRank */
    std::vector<int> ColRanks(const int procRank, const SpGEMM3DParams& params) {

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

    inline IT GetLocNnz() const {return locNnz;}
    inline IT GetLocNcols() const {return locNcols;}
    inline IT GetLocNrows() const {return locNrows;}

    inline float GetGlobDensity() const {return globDensity;}
    inline float GetLocDensity() const {return locDensity;}

    inline SPLIT GetSplit() const {return split;}

    inline std::vector<IT> * GetNnzArr() {return nnzArr;}

    inline std::pair<IT,IT> GetDims3D() {return dims3D;}

private:

    // Global info
    IT nnz;
    IT ncols;
    IT nrows;
    
    // Info about actual 2D grid
    IT locNnz;
    IT locNcols;
    IT locNrows;
    IT locNcolsExact;
    IT locNrowsExact;
    int rowRank; //rank in actual 2d grid
    int colRank; //^^

    float globDensity;
    float locDensity;

    // Row or column split
    SPLIT split;    
    
    // Stores nnz per row/column
    NnzTuples * nnzTuples;

    DER * locMat;

    // Stores nnz per processor in hypothetical 3D grid
    std::vector<IT> * nnzArr;
    std::pair<IT,IT> dims3D;
    

};


}//autotuning
}//combblas


#endif
