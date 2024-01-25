
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
        locNnz(M.seqptr()->getnnz()), locNcols(M.seqptr()->getncol()), locNrows(M.seqptr()->getnrow())
     {

        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        split = M.isColSplit() ? COL_SPLIT : ROW_SPLIT;

#ifdef PROFILE
        auto stime = MPI_Wtime();
#endif

        nnzArr = new distArr(InitNnzVec(M));

#ifdef PROFILE
        auto etime = MPI_Wtime();
        statPtr->Log("nnzArr Init Time: " + std::to_string(etime - stime));
        statPtr->Print("nnzArr Init Time: " + std::to_string(etime - stime));
#endif

        //Synchronize to ensure all processes have activated the nnzArray dist_object
        upcxx::barrier();
        
    }


    /* Initialize global pointer to local nnz array */
    upcxx::global_ptr<IT> InitNnzVec(SpParMat3D<IT,NT,DER>& M) {

        upcxx::global_ptr<IT> globNnzVec(upcxx::new_array<IT>(locNcols));
        auto locNnzVec = globNnzVec.local();

        for (auto colIter = M.seqptr()->begcol(); colIter != M.seqptr()->endcol(); colIter++) {
            locNnzVec[colIter.colid()] = colIter.nnz(); 
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
        
        switch(split) {

            case COL_SPLIT:
            {
                return ComputeLocalNnzColSplit(ppn,nodes,layers);
            }
            case ROW_SPLIT:
            {
                return ComputeLocalNnzRowSplit(ppn,nodes,layers);
            }
            default:
            {
                exit(1);
            }

        }

    }


    IT ComputeLocalNnzColSplit(const int ppn, const int nodes, const int layers) {

        IT locNnz = 0;

        const int totalProcs = ppn*nodes;
        const int cols2D = RoundedSqrt<int,int>(totalProcs);
        const int rows2D = cols2D;

        const int gridSize = totalProcs / layers;
        const int gridRows = RoundedSqrt<int,int>(gridSize);
        const int gridCols = gridRows;

        const int gridRank = rank % gridSize; // rank in grid
        const int gridRowRank = gridRank % gridCols; // rank in processor row of grid
        const int gridColRank = gridRank / gridRows; // rank in processor column of grid
        const int fiberRank = rank / gridSize;
        
        /* Compute columns per processor in 3d grid */
        IT cols3D = ncols / (gridCols * layers); 
        IT firstCol = LocalNcols(totalProcs) * (cols2D/gridCols) * gridRowRank +
                        cols3D * fiberRank; 
        IT lastCol = firstCol + cols3D - 1;

        /* Compute rows per processor in 3d grid */
        IT rows3D = rows2D * (LocalNrows(totalProcs) / gridRows);
        IT firstRow = gridColRank * rows2D; 
        IT lastRow = firstRow + rows2D - 1;


        /* Fetch nnz counts for columns/rows mapped to this processor in the symbolic 3D grid  */
        IT locNnz3D = 0;
        // foreach column
        for (const IT j=firstCol; j<=lastCol; j++) {
            // foreach row block in this column
            for (const IT i = firstRow; i<=lastRow; i+=locNrows) {
                int targetRank = TargetRank(i,j, cols2D);
                locNnz3D += FetchNnz(targetRank, i, j).wait(); //TODO: Can we use a callback + atomics instead?
            }
        }
        
        return locNnz3D;
        
    }


    IT ComputeLocalNnzRowSplit(const int ppn, const int nodes, const int layers) {
        //TODO
    }


    //which rank does the jth row block of column i live on?
    int TargetRank(IT i, IT j, int cols2D) {
        int procRow = i / locNrows;
        int procCol = j / locNcols;
        return procRow * cols2D + procCol; 
    }

    
    upcxx::future<IT> FetchNnz(int targetRank, int i, int j) {
        

        int locJ = j % locNcols;

        return nnzArr->fetch(targetRank).then(
            [locJ](upcxx::global_ptr<IT> nnzPtr) {
                return upcxx::rget(nnzPtr+ locJ);
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

    distArr * nnzArr;

};


}//autotuning
}//combblas


#endif
