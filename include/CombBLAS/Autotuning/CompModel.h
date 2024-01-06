
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


struct RegressionParams {double b; double m;} typedef RegressionParams;

/* Use single-variable regression model T = mx + b  */
class RegressionCompModel : public CompModel {
    

    RegressionCompModel(RegressionParams& p, std::function<long()> ComputeFLOPS):
    b(p.b), m(p.m), ComputeFLOPS(ComputeFLOPS) 
    {
        
    }
    
    double ComputeTime() {

        double x = ComputeFLOPS();
        return m*x + b;

    }

private:
    
    double b; double m;
    std::function<long()> ComputeFLOPS;

};

//computed using numpy.polyfit
RegressionParams localSpGEMMRegressionPerlmutter {1.0, 1.0};

}//autotuning
}//combblas



#endif


