
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

    typedef upcxx::dist_object<std::vector<IT>> distObj;

    enum SPLIT {COL_SPLIT, ROW_SPLIT} typedef SPLIT;


    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& M):
        nnz(M.getnnz()), ncols(M.getncol()), nrows(M.getnrow()),
        locNnz(M.seqptr()->getnnz()), 
        nnzArr(new distObj(std::vector<IT>(0))) {

        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
        split = M.isColSplit() ? COL_SPLIT : ROW_SPLIT;

        //Synchronize to ensure all processes have activated the nnzArray dist_object
        upcxx::barrier();
        
    }

 /* SpGEMM3DMatrixInfo(IT nnz, IT cols, IT rows):
    nnz(nnz), ncols(cols), nrows(rows) {
        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
    }*/


    /* Approximate local nnz using matrix density
     * This actually just computes the avg nnz per processor
     */
    IT ApproxLocalNnzDensity(const int totalProcs) {

        IT localNcols = LocalNcols(totalProcs); //TODO: These should not be member functions, just members
        IT localNrows = LocalNrows(totalProcs);
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(density * localMatSize);
        return localNnzApprox ;
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


        /*  */

        
    }


    IT ComputeLocalNnzRowSplit(const int ppn, const int nodes, const int layers) {
        //TODO
    }


    


    IT ComputeMsgSize(const int locNnz) {
        return locNnz * GetNzvalSize() +
                locNnz * GetIndexSize() +
                (locNnz + 1) * GetIndexSize();
    }

    
    inline IT LocalNcols(int totalProcs) const {return ncols / static_cast<IT>(sqrt(totalProcs));}
    inline IT LocalNrows(int totalProcs) const {return nrows / static_cast<IT>(sqrt(totalProcs));}

    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}

    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}

    inline float GetDensity() const {return density;}

    inline SPLIT GetSplit() const {return split;}

private:


    IT nnz;
    IT ncols;
    IT nrows;

    IT locNnz;

    float density;

    SPLIT split;    

    distObj * nnzArr;

};


}//autotuning
}//combblas


#endif
