
#ifndef PLATFORMPARAMS_H
#define PLATFORMPARAMS_H


#include <exception>


namespace combblas {

namespace autotuning {
/* 
 * TODO: Topology specific params
 * TODO: intrasocket,intranode alpha/beta
 */

class PlatformParams {
public:
    
    //TODO: Provide utilities for measuring members     

    //alpha is us
    //beta is bytes/us
    PlatformParams(float internodeAlpha, float internodeBeta, int coresPerNode, int devsPerNode, int peakGFLOPS) : 
    internodeAlpha(internodeAlpha), internodeBeta(internodeBeta), coresPerNode(coresPerNode), devsPerNode(devsPerNode),
    peakGFLOPS(peakGFLOPS)
    {}
    ~PlatformParams(){}
    
    inline float GetInternodeAlpha() const {return internodeAlpha;} 
    inline float GetInternodeBeta() const {return internodeBeta;} 
    inline int GetCoresPerNode() const {return coresPerNode;}
    inline int GetDevsPerNode() const {return devsPerNode;}
    
    //TODO: Measure local SpGEMM FLOPS and use that instead
    inline int GetPeakGFLOPS() const {return peakGFLOPS;}
    
    float MeasureInternodeAlpha() {throw std::runtime_error("Not implemented");}
    float MeasureInternodeBeta() {throw std::runtime_error("Not implemented");}
    

private:
    
    float internodeAlpha;
    float internodeBeta;

    int coresPerNode;
    int devsPerNode;
    
    int peakGFLOPS;    
    
};

//Values obtained with osu microbenchmarks
// peak = 3.5Ghz * 2 fmadd * 2 pipelines * 8 flops per vector register * 128 cores
PlatformParams perlmutterParams(3.9, 2406.87, 128, 4, (3.5)*2*2*8*128);

} //autotuning
}//combblas
#endif
