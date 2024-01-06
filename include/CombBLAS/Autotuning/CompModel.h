
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
        double timeSeconds = ComputeFLOPS() / (peakFLOPS);
        return timeSeconds * 1e6; //convert to us
    }
    
private:
    long peakFLOPS;
    std::function<long()> ComputeFLOPS;
};

}//autotuning
}//combblas



#endif


