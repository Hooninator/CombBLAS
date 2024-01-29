
#ifndef SPGEMM3DMATRIXINFO_H
#define SPGEMM3DMATRIXINFO_H

#include "common.h"



namespace combblas {
namespace autotuning {
template <typename IT, typename NT, typename DER>
class SpGEMM3DMatrixInfo {

public:

    typedef IT indexType;
    typedef NT nzType;
    typedef DER seqType;

    typedef upcxx::dist_object<upcxx::global_ptr<IT>> distArr;

    enum SPLIT {COL_SPLIT, ROW_SPLIT} typedef SPLIT;

    //NOTE: loc* are values for the actual 2D processor grid
    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& M):
        nnz(M.getnnz()), ncols(M.getncol()), nrows(M.getnrow()),
        locNnz(M.seqptr()->getnnz()), 
        locNcols(M.getncol() / RoundedSqrt<IT,IT>(jobPtr->totalTasks)), 
        locNrows(M.getnrow() / RoundedSqrt<IT,IT>(jobPtr->totalTasks))
     {
        
        INIT_TIMER();

        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        split = M.isColSplit() ? COL_SPLIT : ROW_SPLIT;
        
        START_TIMER();

        nnzArrCol = new distArr(InitNnzArrCol(M));

        END_TIMER("nnzArrCol Init Time: ");

        START_TIMER();

        nnzArrRow = new distArr(InitNnzArrRow(M));
        
        END_TIMER("nnzArrRow Init Time: ");

        //Synchronize to ensure all processes have activated the  dist_object
        upcxx::barrier();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }


    /* Initialize global pointer to local nnz array */
    //TODO: Does this even need to be distributed?
    //TODO: Move this to SpParMat constructor
    upcxx::global_ptr<IT> InitNnzArrCol(SpParMat3D<IT,NT,DER>& M) {

        upcxx::global_ptr<IT> globNnzVec(upcxx::new_array<IT>(locNcols));
        auto locNnzVec = globNnzVec.local();

        //TODO: Use multithreaded versions of this
        for (auto colIter = M.seqptr()->begcol(); colIter != M.seqptr()->endcol(); colIter++) {
            locNnzVec[colIter.colid()] = colIter.nnz(); 
        }

        return globNnzVec;

    } 

    
    upcxx::global_ptr<IT> InitNnzArrRow(SpParMat3D<IT,NT,DER>& M) {
        
        upcxx::global_ptr<IT> globNnzVec(upcxx::new_array<IT>(locNrows));
        auto locNnzVec = globNnzVec.local();

        //JB: This is terrible
        for (auto colIter = M.seqptr()->begcol(); colIter != M.seqptr()->endcol(); colIter++) {
            // Iterate through each element of this column and add it to the nonzero array
            for (auto nzIter = M.seqptr()->begnz(colIter); nzIter != M.seqptr()->endnz(colIter); nzIter++) {
                locNnzVec[nzIter.rowid()] += 1;
            }

        }

        return globNnzVec;

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
        IT locNcols3D = ncols / (gridCols * layers); 
        IT firstCol = locNcols * (procCols2D/gridCols) * gridRowRank +
                        locNcols3D * fiberRank; 
        IT lastCol = firstCol + locNcols3D;

        /* Compute rows per processor in 3d grid */
        IT locNrows3D = nrows / gridRows;
        IT firstRow = gridColRank * locNrows3D; 
        IT lastRow = firstRow + locNrows3D;

        /* Row block offset */
        IT rowOffset = locNrows;

        /* Fetch nnz counts for columns/rows mapped to this processor in the symbolic 3D grid  */
        IT locNnz3D = 0;

        //TODO: Parallelize this loop with openMP
        // foreach column
        IT lrow; IT lcol;
        for (IT j=firstCol; j<lastCol; j++) {
            // foreach row block in this column
            for (IT i = firstRow; i<lastRow; i+=rowOffset) {
                if (i >= lastRow) break;
                int targetRank = TargetRank(i,j, procCols2D);
                locNnz3D += FetchNnz(targetRank, j, locNcols, this->nnzArrCol).wait(); //TODO: Can we use a callback + atomics instead?
            }
        }

        // Handle leftover columns
        for (IT j=(ncols / locNcols3D) * locNcols3D; j<ncols; j++) {
            for (IT i = firstRow; i<lastRow; i+=rowOffset) {
                if (i >= lastRow) break;
                int targetRank = TargetRank(i,j, procCols2D);
                locNnz3D += FetchNnz(targetRank, j, locNcols, this->nnzArrCol).wait(); //TODO: Can we use a callback + atomics instead?
            }
        }


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

        
        IT locNcols3D = ncols / gridCols; 
        IT firstCol = gridRowRank * locNcols3D;
        IT lastCol = firstCol + locNcols3D;

        IT locNrows3D = nrows / (gridCols * layers);
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

        IT locNnz3D = 0;

        // Activate the juice
        
        //TODO: Message aggregation
        //TODO: Should probably avoid fetching rows with no nonzeros
        IT lrow; IT lcol;
        for (IT i = firstRow; i<lastRow; i++) {
            for (IT j = firstCol; j<lastCol; j+=colOffset) {
                if (j >= lastCol) break; //hack for not evenly dividing cols
                int targetRank = TargetRank(i,j, procCols2D);
                locNnz3D += FetchNnz(targetRank, i, locNrows, this->nnzArrRow).wait();
            }
        }

        // Handle leftovers
        for (IT i = (nrows / locNrows3D) * locNrows3D; i<nrows; i++) {
            for (IT j = firstCol; j<lastCol; j+=colOffset) {
                if (j >= lastCol) break; //hack for not evenly dividing cols
                int targetRank = TargetRank(i,j, procCols2D);
                locNnz3D += FetchNnz(targetRank, i, locNrows, this->nnzArrRow).wait();
            }
        }

        DEBUG_LOG("Nnz 3d B: " + std::to_string(locNnz3D));


        return locNnz3D;
    }


    //which rank does the jth row of column i live on?
    //TODO: This fails when the rows/columns are not evenly distributed across the 2D grid
    int TargetRank(IT i, IT j, int procDim2D, IT * lrow, IT * lcol) {

        IT rowsPerProc = nrows / procDim2D;
        IT colsPerProc = ncols / procDim2D;

        int procRow = std::min(i / rowsPerProc, static_cast<IT>(procDim2D-1));
        int procCol = std::min(j / colsPerProc, static_cast<IT>(procDim2D-1));
        
        *lrow = i - procRow*rowsPerProc;
        *lcol = j - procCol*colsPerProc;

        return procRow * procDim2D + procCol; 
    }


    int TargetRank(IT i, IT j, int procDim2D) {
        IT procRow = std::min(i / locNrows, (IT)(procDim2D-1));
        IT procCol = std::min(j / locNcols, (IT)(procDim2D-1));
        return procCol + procRow*procDim2D;
    }

    // idx should be column or row index, depending on how nnzArr is indexed 
    upcxx::future<IT> FetchNnz(int targetRank, IT locIdx, distArr * nnzArr) {
        
        return nnzArr->fetch(targetRank).then(
            [locIdx](upcxx::global_ptr<IT> nnzPtr) {
                return upcxx::rget(nnzPtr + locIdx);
            }
        );

    }

    upcxx::future<IT> FetchNnz(int targetRank, IT idx, IT dim2D, distArr * nnzArr) {
        
        IT locIdx = idx % dim2D;
    
        return nnzArr->fetch(targetRank).then(
            [locIdx](upcxx::global_ptr<IT> nnzPtr) {
                return upcxx::rget(nnzPtr + locIdx);
            }
        );

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

    float density;

    SPLIT split;    

    distArr * nnzArrCol;
    distArr * nnzArrRow;

};


}//autotuning
}//combblas


#endif
