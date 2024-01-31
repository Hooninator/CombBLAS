
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
        locNrows(M.getnrow() / RoundedSqrt<IT,IT>(jobPtr->totalTasks)),
        locNcolsExact(M.seqptr()->getncol()),
        locNrowsExact(M.seqptr()->getnrow())
     {
        
        INIT_TIMER();

        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        split = M.isColSplit() ? COL_SPLIT : ROW_SPLIT;
        
        START_TIMER();

        nnzArrCol = NnzArrCol(M); 

        END_TIMER("nnzArrCol Init Time: ");

        START_TIMER();

        END_TIMER("nnzArrRow Init Time: ");

        //Synchronize to ensure all processes have activated the  dist_object
        upcxx::barrier();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }


    /* Initialize global pointer to local nnz array */
    //TODO: Does this even need to be distributed?
    //TODO: Move this to SpParMat constructor
    upcxx::global_ptr<IT> InitNnzArrCol(SpParMat3D<IT,NT,DER>& M) {

        upcxx::global_ptr<IT> globNnzVec(upcxx::new_array<IT>(locNcolsExact));
        auto locNnzVec = globNnzVec.local();

        std::memset((void*)locNnzVec, 0, sizeof(IT)*locNcolsExact); //for some reason, this is necessary

        //TODO: Use multithreaded versions of this
        for (auto colIter = M.seqptr()->begcol(); colIter != M.seqptr()->endcol(); colIter++) {
            locNnzVec[colIter.colid()] = colIter.nnz(); 
        }

        return globNnzVec;

    } 


    /* Initialize array containing nnz per column on each processor, then gather on processor 0  */
    std::vector<IT> NnzArrCol(SpParMat3D<IT,NT,DER>& M) {
        
        // Make local vector, init to zero
        std::vector<IT> nnzArrLoc(locNcolsExact);
        std::fill(nnzArrLoc.begin(), nnzArrLoc.end(), 0);
        
        // Init local columns
        std::for_each(M.seqptr()->begcol, M.seqptr()->endcol(), [&nnzArrLoc](auto colIter) mutable {
            nnzArrLoc[colIter.colid()] = colIter.nnz();
        });

        // Reduce across each processor column to get column counts on processor row 0
        std::vector<IT> nnzArrAggregate(locNcolsExact);
        MPI_Reduce(nnzArrLoc.data(), nnzArrAggregate.data(), nnzArrLoc.size(),
                    MPIType<IT>(), M.getcommgrid()->GetColWorld());

        //Now, gatherv onto rank 0 across row 0 of processor grid
        //Note that gatherv is needed instead of gather because the last processor could have edge columns

        // Vector storing nonzeros of all ranks 
        std::vector<IT> nnzArrGlob(M.ncols);

        // Get recvcounts
        std::vector<IT> recvCounts(worldSize);
        IT locRecvCount = locNcols;
        MPI_Gather((void*)(&locRecvCount), 1, MPIType<IT>(), 
                        recvCounts.data(), 1, MPIType<IT>(), 
                        0, M.getcommgrid()->GetRowWorld());

        // Get displacements
        // This should be the same as recvcounts, since we want to displace each received array
        // by its size
        std::vector<IT> * displs = &recvCounts;

        // Gatherv to populate rank 0 array for all columns
        MPI_Gatherv(nnzArrLoc.data(), nnzArrLoc.size(), MPIType<IT>(),
                    nnzArrGlob.data(), recvCounts.data(), displs->data(), MPIType<IT>(),
                    0, M.getcommgrid()->GetRowWorld());

        return nnzArrGlob;

    }

    /* Initialize array containing nnz per row on each processor, then gather on processor 0 */
    std::vector<IT> NnzArrRow(SpParMat3D<IT,NT,DER>& M) {
    
    }

    
    upcxx::global_ptr<IT> InitNnzArrRow(SpParMat3D<IT,NT,DER>& M) {
        
        upcxx::global_ptr<IT> globNnzVec(upcxx::new_array<IT>(locNrowsExact));
        auto locNnzVec = globNnzVec.local();

        std::memset((void*)locNnzVec,0,sizeof(IT)*locNrowsExact);

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
     //JB: This is impractically slow as it currently stands...
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

        upcxx::barrier();

        END_TIMER("Compute 3D nnz time: ");

        return locNnz;

    }


    IT ComputeLocalNnzColSplit(const int ppn, const int nodes, const int layers) {

        const int totalProcs = ppn*nodes;

        if (totalProcs==1) return locNnz;
        if (rank>=totalProcs) return 0; //return if this rank isn't part of the 3D grid
        bool useConjFut = DecideConjFut(ppn,nodes,layers);

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
        //if (gridSize==1) lastRow = (nrows / layers) * layers;

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
                                        procCols2D, useConjFut);


        DEBUG_LOG("Nnz 3d A: " + std::to_string(locNnz3D));

        return locNnz3D;
        
    }


    IT ComputeLocalNnzRowSplit(const int ppn, const int nodes, const int layers) {

        const int totalProcs = ppn*nodes;

        if (totalProcs==1) return locNnz;
        if (rank>=totalProcs) return 0; //return if this rank isn't part of the 3D grid

        bool useConjFut = DecideConjFut(ppn, nodes, layers);

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
                                        procRows2D, useConjFut); 

        DEBUG_LOG("Nnz 3d B: " + std::to_string(locNnz3D));


        return locNnz3D;
    }

    
    IT SumLocalNnzRange(SPLIT split, 
                        IT firstRow, IT lastRow,
                        IT firstCol, IT lastCol,
                        int procDim2D,
                        bool useConjFut) 
    {
        IT locNnz3D = 0;
        auto addNnz = [&locNnz3D](IT colNnz)mutable{locNnz3D+=colNnz;};
        
        IT outerStart;
        IT outerEnd;
        IT innerStart;
        IT innerEnd;
        IT offset;
        IT locDimMod;
        distArr * nnzArr;
        
        if (split==ROW_SPLIT) {
            outerStart = firstRow;
            outerEnd = lastRow;
            innerStart = firstCol;
            innerEnd = lastCol;
            offset = this->locNcols;
            locDimMod = this->locNrows;
            nnzArr = this->nnzArrRow;
        } else if (split==COL_SPLIT) {
            outerStart = firstCol;
            outerEnd = lastCol;
            innerStart = firstRow;
            innerEnd = lastRow;
            offset = this->locNrows;
            locDimMod = this->locNcols;
            nnzArr = this->nnzArrCol;
        }

#ifdef DEBUG
        std::map<int,int> targets;
#endif

        upcxx::future<> fetchFuts = upcxx::make_future();

        //TODO: Message aggregation
        //TODO: Should probably avoid fetching rows with no nonzeros
        
        for (IT k = outerStart; k<outerEnd; k++) {
            for (IT l = innerStart; l<innerEnd; l+=offset) {
                int targetRank = TargetRank(k,l, procDim2D, split);
#ifdef DEBUG
                targets.emplace(targetRank, 0);
                targets[targetRank] += 1;
#endif
                if (useConjFut) {
                    upcxx::future<> fetchFut = FetchNnz(targetRank, k, locDimMod, nnzArr).then(
                        addNnz
                    );
                    fetchFuts = upcxx::when_all(fetchFuts, fetchFut);
                } else {
                    locNnz3D += FetchNnz(targetRank, k, locDimMod, nnzArr).wait();
                }
            }
        }
        
        //Handle edge case
        bool edge;
        IT edgeEnd;
        if (split==ROW_SPLIT) {
            edge = (lastRow==(nrows / procDim2D)*procDim2D);
            edgeEnd = nrows;
        } else if (split==COL_SPLIT) {
            edge = (lastCol==(ncols / procDim2D)*procDim2D);
            edgeEnd = ncols;
        }   

        if (edge) {
            for (IT k = outerEnd; k<edgeEnd; k++) {
                for (IT l = innerStart; l<outerStart; l+=offset) {
                    int targetRank = TargetRank(k,l, procDim2D, split);
#ifdef DEBUG
                    targets.emplace(targetRank, 0);
                    targets[targetRank] += 1;
#endif
                    if (useConjFut) {
                        upcxx::future<> fetchFut = FetchNnz(targetRank, k, locDimMod, nnzArr).then(
                            addNnz
                        );
                        fetchFuts = upcxx::when_all(fetchFuts, fetchFut);
                    } else {
                        locNnz3D += FetchNnz(targetRank, k, locDimMod, nnzArr).wait();
                    }
                }
            }
        }

        if (useConjFut) fetchFuts.wait();

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


    upcxx::future<IT> FetchNnz(int targetRank, IT idx, IT dim2D, distArr * nnzArr) {
        IT locIdx = idx % dim2D;
        return nnzArr->fetch(targetRank).then(
            [locIdx](upcxx::global_ptr<IT> nnzPtr) {
                return upcxx::rget(nnzPtr + locIdx);
            }
        );
    }


    bool DecideConjFut(int ppn, int nodes, int layers) {
        //TODO:
        return true;
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

    std::vector<IT> nnzArrCol;
    std::vector<IT> nnzArrRow;

};


}//autotuning
}//combblas


#endif
