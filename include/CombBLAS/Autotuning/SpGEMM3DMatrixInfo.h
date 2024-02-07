
#ifndef SPGEMM3DMATRIXINFO_H
#define SPGEMM3DMATRIXINFO_H

#include "common.h"


#define NNZ_THRESH 0

namespace combblas {
namespace autotuning {
template <typename IT, typename NT, typename DER>
class SpGEMM3DMatrixInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;

    typedef SpDCCols<IT,IT> NnzMat;

    enum SPLIT {COL_SPLIT, ROW_SPLIT} typedef SPLIT;

    //NOTE: loc* are values for the actual 2D processor grid
    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& Mat):
        nnz(Mat.getnnz()), ncols(Mat.getncol()), nrows(Mat.getnrow()),
        locNnz(Mat.seqptr()->getnnz()), 
        locNcols(Mat.getncol() / RoundedSqrt<IT,IT>(worldSize)), 
        locNrows(Mat.getnrow() / RoundedSqrt<IT,IT>(worldSize)),
        locNcolsExact(Mat.seqptr()->getncol()),
        locNrowsExact(Mat.seqptr()->getnrow())
     {
        
        INIT_TIMER();

        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        split = Mat.isColSplit() ? COL_SPLIT : ROW_SPLIT;
        
        START_TIMER();

#ifdef DEBUG
        debugPtr->Log("Starting matcol");
#endif

        nnzMatCol = NnzMatCol(Mat); 

#ifdef DEBUG
        debugPtr->Log("Done with matcol");
#endif

        END_TIMER("nnzMatCol Init Time: ");

        START_TIMER();

        nnzMatRow = NnzMatRow(Mat);

        END_TIMER("nnzMatRow Init Time: ");

        
    }



    /* Create sparse matrix storing nnz for each block row of each column and distribute across all ranks  */
    NnzMat * NnzMatCol(SpParMat3D<IT,NT,DER>& Mat) {

        INIT_TIMER();

        START_TIMER();

        std::tuple<IT,IT,IT> * locTuples = new std::tuple<IT,IT,IT>[locNcolsExact];

        int colRank = Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcCol();
        int rowRank = Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcRow();
        
        // Init local data
        int locTupleSize = 0;
        for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
            if (colIter.nnz()>NNZ_THRESH) {
                locTuples[locTupleSize] = std::tuple<IT,IT,IT>{colRank,  colIter.colid() + locNcols*rowRank, colIter.nnz()}; 
                locTupleSize++;
            }
        }

        END_TIMER("Time for local tuple construction in col mat: ");

#ifdef DEBUG
        debugPtr->Log("locNnzTuples col");
        for (int i=0; i<locTupleSize; i++) {
            debugPtr->Log(std::to_string(i) + ":" + TupleStr(locTuples[i]));
        }
#endif

        START_TIMER();

        NnzMat * nnzMat = new NnzMat(Mat.getcommgrid()->GetCommGridLayer()->GetGridRows(),
                                Mat.getncol(),
                                locTupleSize,
                                locTuples, false);

        END_TIMER("Time for SpMatCol construction: ");

        return nnzMat;

    }


    /* Initialize array containing nnz per row on each processor, then gather on processor 0 */
    NnzMat * NnzMatRow(SpParMat3D<IT,NT,DER>& Mat) {

        INIT_TIMER();

        START_TIMER();

        int colRank = Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcCol();
        int rowRank = Mat.getcommgrid()->GetCommGridLayer()->GetRankInProcRow();

        // JB: I can't figure out a way to avoid mutating nnz during iteration, so we can't just use std::tuple
        std::map<std::tuple<IT,IT>, IT> nnzMap;
        for (auto colIter = Mat.seqptr()->begcol(); colIter != Mat.seqptr()->endcol(); colIter++) {
            for (auto nzIter = Mat.seqptr()->begnz(colIter); nzIter!=Mat.seqptr()->endnz(colIter); nzIter++) {
                std::tuple<IT,IT> t{nzIter.rowid() + locNrows*colRank, rowRank};
                nnzMap.emplace(t, 0);
                nnzMap[t] += 1;
            }
        }

        END_TIMER("Time for local nnzMap construction in row mat: ");

        START_TIMER();

        std::tuple<IT,IT,IT> * locTuples = new std::tuple<IT,IT,IT>[locNrowsExact];

        int locTupleSize = 0;
        /*for (auto const& elem : nnzMap) {
            std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
            locTuples[locTupleSize] = t;
            locTupleSize++;
        }*/

        std::for_each(nnzMap.begin(), nnzMap.end(), 
            [&locTuples, &locTupleSize](auto& elem) mutable {
                std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
                locTuples[locTupleSize] = t;
                locTupleSize++;
            }
        );
        
        END_TIMER("Time for local tuple construction in row mat: ");

#ifdef DEBUG
        debugPtr->Log("locNnzTuples row");
        for (int i=0; i<locTupleSize; i++) {
            debugPtr->Log(std::to_string(i) + ":" + TupleStr(locTuples[i]));
        }
#endif

        START_TIMER();

        NnzMat * nnzMat = new NnzMat(Mat.getnrow(),
                                    Mat.getcommgrid()->GetCommGridLayer()->GetGridRows(),
                                    locTupleSize,
                                    locTuples, false);
        
        END_TIMER("Time for SpMatRow Construction: ");

        return nnzMat;

    }



    /* Approximate local nnz using matrix density
     * This actually just computes the avg nnz per processor
     */
    IT ApproxLocalNnzDensity(const int totalProcs) {

        IT localNcols = LocalNcols(totalProcs); //TODO: These should not be member functions, just members
        IT localNrows = LocalNrows(totalProcs);
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(density * localMatSize);
        return localNnzApprox;
    }

    
    /* Given local nnz in initial 2D processor grid, compute nnz per processor in 3D processr grid
     * WITHOUT explicitly forming the 3D processor grid. */
    IT ComputeLocalNnz(const int ppn, const int nodes, const int layers) {
        
        INIT_TIMER();

        START_TIMER();

        IT locNnz = 0;

        switch(split) {

            case COL_SPLIT:
            {
                locNnz = ComputeLocalNnzColSplit(ppn,nodes,layers);
                break;
            }
            case ROW_SPLIT:
            {
                
                locNnz = ComputeLocalNnzRowSplit(ppn,nodes,layers);
                break;
            }
            default:
            {
                exit(1);
            }

        }


        END_TIMER("Compute 3D nnz time: ");

        return locNnz;

    }

    
    std::vector<IT> ComputeNnzArr(const int ppn, const int nodes, const int layers) {
        
        INIT_TIMER();

        START_TIMER();

        std::vector<IT> nnzArr;

        switch(split) {

            case COL_SPLIT:
            {
                nnzArr = NnzArrColSplit(ppn,nodes,layers);
                break;
            }
            case ROW_SPLIT:
            {
                nnzArr = NnzArrRowSplit(ppn, nodes, layers);
                break;
            }
            default:
            {
                exit(1);
            }

        }


        END_TIMER("Compute 3D nnz time: ");

        return nnzArr;

    }

    std::vector<IT> NnzArrColSplit(const int ppn, const int nodes, const int layers) {

        const int totalProcs = ppn*nodes;

        std::vector<IT> nnzArr(totalProcs);

        // Local nnz array
        for (auto colIter = nnzMatCol->begcol(); colIter!=nnzMatCol->endcol(); colIter++) {
            int j = colIter.colid();
            for (auto nzIter = nnzMatCol->begnz(colIter); nzIter!=nnzMatCol->endnz(colIter); nzIter++) {
                int i = nzIter.rowid();
                int owner = GetOwner3D(ppn, nodes, layers, i*locNrows, j, COL_SPLIT);
                nnzArr[owner] += nzIter.value();
            }
        }


        // Allreduce to get complete counts for each process
        MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr.data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
        debugPtr->LogVecSameLine(nnzArr, std::string{"nnzArr A: "});
#endif

        return nnzArr;

    }
        

    std::vector<IT> NnzArrRowSplit(const int ppn, const int nodes, const int layers) {

        const int totalProcs = nodes*ppn;
        std::vector<IT> nnzArr(totalProcs);

        // Local data
        for (auto colIter = nnzMatRow->begcol(); colIter!=nnzMatRow->endcol(); colIter++) {
            int j = colIter.colid();
            for (auto nzIter = nnzMatRow->begnz(colIter); nzIter!=nnzMatRow->endnz(colIter); nzIter++) {
                int i = nzIter.rowid();
                int owner = GetOwner3D(ppn, nodes, layers, i, j*locNcols, ROW_SPLIT);
                nnzArr[owner] += nzIter.value();
            }
        }

        // Allreduce to sum all nnz
        MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr.data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
        debugPtr->LogVecSameLine(nnzArr, std::string{"nnzArr B: "});
#endif

        return nnzArr;

    }

    int GetOwner3D(const int ppn, const int nodes, const int layers, const int i, const int j, SPLIT split) {

        /* Info about currently, actually formed 2D processor grid */
        const int totalProcs2D = jobPtr->totalTasks;
        const int procCols2D = RoundedSqrt<int,int>(totalProcs2D);
        const int procRows2D = procCols2D;

        /* Info about 3D grid */
        const int totalProcs = ppn*nodes;
        const int gridSize = totalProcs / layers;
        const int gridRows = RoundedSqrt<int,int>(gridSize);
        const int gridCols = gridRows;

        IT locNcols3D;
        IT locNrows3D;
        IT colDiv;
        IT rowDiv;
        IT layerDiv;
        int layerIdx;
        
        if (split==COL_SPLIT) {
            locNcols3D = locNcols * (procCols2D/gridCols)/layers;
            locNrows3D = locNrows * (procRows2D/gridRows);
            colDiv = locNcols3D*layers;
            rowDiv = locNrows3D;
            layerDiv = locNcols3D;
            layerIdx = j;
        } else if (split==ROW_SPLIT) {
            locNcols3D = locNcols * (procCols2D/gridCols);
            locNrows3D = locNrows * (procRows2D/gridRows)/layers;
            colDiv = locNcols3D;
            rowDiv = locNrows3D*layers;
            layerDiv = locNrows3D;
            layerIdx = i;
        }

        const int prow = std::min(static_cast<IT>(i / rowDiv), static_cast<IT>(gridRows-1));
        const int pcol = std::min(static_cast<IT>(j / colDiv), static_cast<IT>(gridCols-1));
        const int player = std::min(static_cast<IT>((layerIdx / layerDiv)%layers), static_cast<IT>(layers-1));

#ifdef DEBUG
        debugPtr->Log("locNcols3D: " + std::to_string(locNcols3D));
        debugPtr->Log("locNrows3D: " + std::to_string(locNrows3D));
        debugPtr->Log("(i: " + std::to_string(i) + ", j: " + std::to_string(j) + ")");
        debugPtr->Log("(prow: " + std::to_string(prow) + ", pcol: " + std::to_string(pcol) + ", player: " + std::to_string(player) + ")");
#endif

        return (pcol + prow*gridCols + player*gridSize);

    }


    IT ComputeLocalNnzColSplit(const int ppn, const int nodes, const int layers) {
        
        const int totalProcs = ppn*nodes;

        if (totalProcs==1) return locNnz;
        if (rank>=totalProcs) return 0; //return if this rank isn't part of the 3D grid

        /* Info about currently, actually formed 2D processor grid */
        const int totalProcs2D = jobPtr->totalTasks;
        const int procCols2D = RoundedSqrt<int,int>(totalProcs2D);
        const int procRows2D = procCols2D;

        const int gridSize = totalProcs / layers;
        const int gridRows = RoundedSqrt<int,int>(gridSize);
        const int gridCols = gridRows;

        const int gridRank = rank % gridSize; // rank in grid
        const int gridRowRank = gridRank % gridCols; // rank in processor row of grid
        const int gridColRank = gridRank / gridRows; // rank in processor column of grid
        const int fiberRank = rank / gridSize;
        
        /* Compute columns per processor in 3d grid */
        IT locNcols3D;
        locNcols3D = locNcols * (procCols2D/gridCols); 
        if (layers!=1) locNcols3D = locNcols3D / layers;

        IT firstCol = locNcols * (procCols2D/gridCols) * gridRowRank +
                        locNcols3D * fiberRank; 
        IT lastCol = firstCol + locNcols3D;

        /* Compute rows per processor in 3d grid */
        IT locNrows3D = locNrows * (procRows2D/gridRows);
        IT firstRow = gridColRank * locNrows3D; 
        IT lastRow = firstRow + locNrows3D;

        /* Row block offset */
        IT rowOffset = locNrows;

        DEBUG_LOG("Rows 3D: " + std::to_string(locNrows3D));
        DEBUG_LOG("Cols 3D: " + std::to_string(locNcols3D));
        DEBUG_LOG("First col: " + std::to_string(firstCol));
        DEBUG_LOG("Last col: " + std::to_string(lastCol));
        DEBUG_LOG("First row: " + std::to_string(firstRow));
        DEBUG_LOG("Last row: " + std::to_string(lastRow));
        DEBUG_LOG("Local rows: " + std::to_string(locNrows));
        DEBUG_LOG("Local cols: " + std::to_string(locNcols));
        DEBUG_LOG("Local rows exact: " + std::to_string(locNrowsExact));
        DEBUG_LOG("Local cols exact: " + std::to_string(locNcolsExact));

        IT locNnz3D = SumLocalNnzRange(COL_SPLIT, firstRow, lastRow,
                                        firstCol, lastCol,
                                        procCols2D);


        DEBUG_LOG("Nnz 3d A: " + std::to_string(locNnz3D));

        return locNnz3D;
        
    }


    IT ComputeLocalNnzRowSplit(const int ppn, const int nodes, const int layers) {

        const int totalProcs = ppn*nodes;

        if (totalProcs==1) return locNnz;
        if (rank>=totalProcs) return 0; //return if this rank isn't part of the 3D grid

        /* Info about currently, actually formed 2D processor grid */
        const int totalProcs2D = jobPtr->totalTasks;
        const int procCols2D = RoundedSqrt<int,int>(totalProcs2D);
        const int procRows2D = procCols2D;

        const int gridSize = totalProcs / layers;
        const int gridRows = RoundedSqrt<int,int>(gridSize);
        const int gridCols = gridRows;

        const int gridRank = rank % gridSize; // rank in grid
        const int gridRowRank = gridRank % gridCols; // rank in processor row of grid
        const int gridColRank = gridRank / gridRows; // rank in processor column of grid
        const int fiberRank = rank / gridSize;

        IT locNcols3D = locNcols * (procCols2D/gridCols); 
        IT firstCol = gridRowRank * locNcols3D;
        IT lastCol = firstCol + locNcols3D;

        IT locNrows3D; 
        locNrows3D = locNrows * (procRows2D/gridRows);
        if (layers!=1) locNrows3D /= layers;

        IT firstRow = gridColRank * (procRows2D / gridRows) * locNrows +
                        locNrows3D * fiberRank;
        IT lastRow = firstRow + locNrows3D;

        IT colOffset = locNcols;

        DEBUG_LOG("Rows 3D: " + std::to_string(locNrows3D));
        DEBUG_LOG("Cols 3D: " + std::to_string(locNcols3D));
        DEBUG_LOG("First col: " + std::to_string(firstCol));
        DEBUG_LOG("Last col: " + std::to_string(lastCol));
        DEBUG_LOG("First row: " + std::to_string(firstRow));
        DEBUG_LOG("Last row: " + std::to_string(lastRow));
        DEBUG_LOG("Local rows: " + std::to_string(locNrows));
        DEBUG_LOG("Local cols: " + std::to_string(locNcols));
        DEBUG_LOG("Local rows exact: " + std::to_string(locNrowsExact));
        DEBUG_LOG("Local cols exact: " + std::to_string(locNcolsExact));

        IT locNnz3D = SumLocalNnzRange(ROW_SPLIT, firstRow, lastRow,
                                        firstCol, lastCol,
                                        procRows2D); 

        DEBUG_LOG("Nnz 3d B: " + std::to_string(locNnz3D));


        return locNnz3D;
    }

    
    IT SumLocalNnzRange(SPLIT split, 
                        IT firstRow, IT lastRow,
                        IT firstCol, IT lastCol,
                        int procDim2D)
    {

        IT locNnz3D = 0;
        auto addNnz = [&locNnz3D](IT colNnz)mutable{locNnz3D+=colNnz;};
        
        IT outerStart;
        IT outerEnd;
        IT innerStart;
        IT innerEnd;
        IT offset;
        IT locDimMod;
        NnzMat * nnzMat;
        
        if (split==ROW_SPLIT) {
            outerStart = firstRow;
            outerEnd = lastRow;
            innerStart = firstCol;
            innerEnd = lastCol;
            offset = this->locNcols;
            locDimMod = this->locNrows;
            nnzMat = this->nnzMatRow;
        } else if (split==COL_SPLIT) {
            outerStart = firstCol;
            outerEnd = lastCol;
            innerStart = firstRow;
            innerEnd = lastRow;
            offset = this->locNrows;
            locDimMod = this->locNcols;
            nnzMat = this->nnzMatCol;
        }

        
        bool edge = false;
        
        // Find index of first local column with nonzeros
        IT locColOffset = 0;
        for (auto colIter = nnzMat->begcol(); colIter != nnzMat->endcol(); colIter++) {
            if (colIter.colid()>=outerStart) {
                break;
            }
            locColOffset++;
        }

        // Find edge case starting point
        IT edgeColOffset = locColOffset;
        for (auto colIter = nnzMat->begcol()+locColOffset; colIter != nnzMat->endcol(); colIter++) {
            if (colIter.colid()>=outerEnd) {
                edge = true;
                break;
            }
            edgeColOffset++;
        }

        // Main loop
        for (auto colIter = nnzMat->begcol() + locColOffset; colIter.colid() < outerEnd; colIter++) {
            for (auto nzIter = nnzMat->begnz(colIter) + (innerStart / offset); (nzIter.rowid() * offset) < innerEnd; nzIter++) {
                locNnz3D += nzIter.value();
            }
        }


        // Handle edge case
        if (edge) {
            for (auto colIter = nnzMat->begcol() + edgeColOffset; colIter != nnzMat->endcol(); colIter++) {
                for (auto nzIter = nnzMat->begnz(colIter) + (innerStart / offset); nzIter != nnzMat->endnz(colIter); nzIter++) {
                    locNnz3D += nzIter.value();
                }
            }
        }

        return locNnz3D;
    }


    //which rank does the jth row of column i live on?
    int TargetRank(IT i, IT j, int procDim2D, IT * lrow, IT * lcol) {

        IT rowsPerProc = nrows / procDim2D;
        IT colsPerProc = ncols / procDim2D;

        int procRow = std::min(i / rowsPerProc, static_cast<IT>(procDim2D-1));
        int procCol = std::min(j / colsPerProc, static_cast<IT>(procDim2D-1));
        
        *lrow = i - procRow*rowsPerProc;
        *lcol = j - procCol*colsPerProc;

        return procRow * procDim2D + procCol; 
    }


    int TargetRank(IT outerDim, IT innerDim, int procDim2D, SPLIT split) {

        int i, j;
        if (split==ROW_SPLIT) {
            i = outerDim;
            j = innerDim;
        } else if (split==COL_SPLIT) {
            i = innerDim;
            j = outerDim;
        }
        
        IT procRow = std::min(i / locNrows, (IT)(procDim2D-1));
        IT procCol = std::min(j / locNcols, (IT)(procDim2D-1));
        return procCol + procRow*procDim2D;
    }



    IT ComputeMsgSize(const int locNnz) {
        return locNnz * GetNzvalSize() +
                locNnz * GetIndexSize() +
                (locNnz + 1) * GetIndexSize();
    }

    
    //NOTE: These compute local sizes for a hypothetical 2D grid
    inline IT LocalNcols(int totalProcs) const {return ncols / static_cast<IT>(sqrt(totalProcs));}
    inline IT LocalNrows(int totalProcs) const {return nrows / static_cast<IT>(sqrt(totalProcs));}

    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}

    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}

    inline IT GetLocNnz() const {return locNnz;}
    inline IT GetLocNcols() const {return locNcols;}
    inline IT GetLocNrows() const {return locNrows;}

    inline float GetDensity() const {return density;}

    inline SPLIT GetSplit() const {return split;}



private:


    IT nnz;
    IT ncols;
    IT nrows;

    IT locNnz;
    IT locNcols;
    IT locNrows;
    IT locNcolsExact;
    IT locNrowsExact;

    float density;

    SPLIT split;    

    NnzMat * nnzMatCol;
    NnzMat * nnzMatRow;

};


}//autotuning
}//combblas


#endif
