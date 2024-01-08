
#ifndef COMPMODEL_H
#define COMPMODEL_H

#include "common.h"

namespace combblas {
namespace autotuning {


class CompModel {

public:
    CompModel(){}
    
    virtual double ComputeTime() {throw std::runtime_error("This method should never be called");}


};

/* T = FLOPS / PeakFLOPS*/
class PeakCompModel : public CompModel {

public:
    
    //FLOPS/sec
    PeakCompModel(long peakFLOPS, std::function<long()> ComputeFLOPS):
    peakFLOPS(peakFLOPS), ComputeFLOPS(ComputeFLOPS) 
    {
    }
    
    double ComputeTime() {
#ifdef DEBUG
        debugPtr->Print("Flops: " + std::to_string(ComputeFLOPS()));
        debugPtr->Print("Peak: " + std::to_string(peakFLOPS));
#endif
        double timeSeconds = static_cast<double>(ComputeFLOPS()) / static_cast<double>((peakFLOPS));
        return timeSeconds * 1e6; //convert to us
    }
    
private:
    long peakFLOPS;
    std::function<long()> ComputeFLOPS;
};


/* T = P(x), P(x) = \sum_{i=0}^d(a*x^i) */
class RegressionCompModel : public CompModel {
    
public:

    RegressionCompModel(std::vector<double>& coeffs, std::function<long()> ComputeFLOPS):
    coeffs(coeffs), ComputeFLOPS(ComputeFLOPS) 
    {
        
    }
    
    double ComputeTime() {

        long x = ComputeFLOPS();
        int i=0;

        double timeSeconds = std::accumulate(coeffs.begin(), coeffs.end(), 0.0, [=](double sum, double a)mutable {
            return sum + a*std::pow(x,i++);
        });
        
        return timeSeconds * 1e6; //convert to us

    }
    
    inline int Degree() const {return coeffs.size();}

private:
    
    std::vector<double> coeffs;
    std::function<long()> ComputeFLOPS;

};

//computed using numpy least squares fitting functionality
//TODO: C++ program to do this
//This is going to make things complicated...
std::vector<double> regSpGEMMPerlmutter {1.41575143, 8.06541948e-07};


}//autotuning
}//combblas



#endif


