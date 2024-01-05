
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

    SpGEMM3DMatrixInfo(SpParMat3D<IT,NT,DER>& M):
    nnz(M.getnnz()), ncols(M.getncol()), nrows(M.getnrow()) {
        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
    }

    SpGEMM3DMatrixInfo(IT nnz, IT cols, IT rows):
    nnz(nnz), ncols(cols), nrows(rows) {
        density = static_cast<float>(nnz) / static_cast<float>(ncols*nrows);
    }

    inline int GetIndexSize() const {return sizeof(IT);}
    inline int GetNzvalSize() const {return sizeof(NT);}

    inline IT GetNnz() const {return nnz;}
    inline IT GetNcols() const {return ncols;}
    inline IT GetNrows() const {return nrows;}

    inline float GetDensity() const {return density;}

private:


    IT nnz;
    IT ncols;
    IT nrows;

    float density;

};


}//autotuning
}//combblas


#endif
