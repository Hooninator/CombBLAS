
#ifndef LOCALSPGEMMMODEL_H
#define LOCALSPGEMMMODEL_H

#include "common.h"
#include "SpGEMM3DMatrixInfo.h"

namespace combblas {
namespace autotuning {


struct LocalSpGEMMInfo {
    
    long long FLOPS;

};

class LocalSpGEMMModel {

public:
    LocalSpGEMMModel(){}
    
    virtual double ComputeTime(LocalSpGEMMInfo * info) {INVALID_CALL_ERR();}


    /* Approximate local FLOPS using density-based nnz estimation */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    long long ApproxLocalMultFLOPSDensity(SpGEMM3DMatrixInfo<AIT, ANT, ADER>& Ainfo, SpGEMM3DMatrixInfo<BIT, BNT, BDER>& Binfo,
                                         int totalProcs,  int gridSize){

        const int layers = totalProcs / gridSize;

        long long tileFLOPS = Ainfo.GetDensity() * (Ainfo.GetNrows() / RoundedSqrt<int,int>(gridSize)) * // estimate nnz per col of A
                        Binfo.GetDensity() * (Binfo.GetNrows() / (layers * RoundedSqrt<int,int>(gridSize))) * // estimate nnz per col of B
                        (Binfo.GetNcols() / RoundedSqrt<int,int>(gridSize)); // once per col of B
        long long localFLOPS = tileFLOPS * RoundedSqrt<int,int>(gridSize); //we do sqrt(gridSize) local multiplies

#ifdef PROFILE
        statPtr->Log("Local FLOPS " + std::to_string(localFLOPS));
#endif

        return localFLOPS;

    }


};


/* T = P(x), P(x) = \sum_{i=0}^d(a*x^i) */
class RegressionLocalSpGEMMModel : public LocalSpGEMMModel {
    
public:

    RegressionLocalSpGEMMModel(std::vector<double>& coeffs):
    coeffs(coeffs) 
    {
        
    }
    
    double ComputeTime(LocalSpGEMMInfo * info) {

        int i=0;

        double timeSeconds = std::accumulate(coeffs.begin(), coeffs.end(), 0.0, [=](double sum, double a)mutable {
            return sum + a*std::pow(info->FLOPS,i++);
        });
        
        return timeSeconds * 1e6; //convert to us

    }
    
    inline int Degree() const {return coeffs.size();}

private:
    
    std::vector<double> coeffs;

};

//TODO: Multivariate Regression Model

//computed using numpy least squares fitting functionality
//TODO: C++ program to do this
//This is going to make things complicated...
std::vector<double> regSpGEMMPerlmutter {1.37959216, 3.96351051e-07};


}//autotuning
}//combblas



#endif


