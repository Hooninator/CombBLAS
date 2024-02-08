
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
    //all memory things are bytes/us
    PlatformParams(float internodeAlpha, float internodeBeta, float intranodeBeta, 
                    int coresPerNode, int devsPerNode, 
                    long peakFLOPS, long memBW): 
        internodeAlpha(internodeAlpha), internodeBeta(internodeBeta), intranodeBeta(intranodeBeta), 

        coresPerNode(coresPerNode), devsPerNode(devsPerNode),

        peakFLOPS(peakFLOPS), memBW(memBW)
    {}
    ~PlatformParams(){}
    
    inline float GetInternodeAlpha() const {return internodeAlpha;} 
    inline float GetInternodeBeta() const {return internodeBeta;} 
    inline float GetIntranodeBeta() const {return intranodeBeta;}
    inline int GetCoresPerNode() const {return coresPerNode;}
    inline int GetDevsPerNode() const {return devsPerNode;}
    
    //TODO: Measure local SpGEMM FLOPS and use that instead
    inline long GetPeakFLOPS() const {return peakFLOPS;}

    inline long GetMemBW() const {return memBW;}
    
    float MeasureInternodeAlpha() {throw std::runtime_error("Not implemented");}
    float MeasureInternodeBeta() {throw std::runtime_error("Not implemented");}

    

private:
    
    float internodeAlpha;
    float internodeBeta;
    
    float intranodeBeta;

    int coresPerNode;
    int devsPerNode;
    
    long peakFLOPS; //peak flops on a SINGLE core 
    long memBW;
    
};

//Values obtained with osu microbenchmarks
// peak = 3.5Ghz * 2 fmadd * 2 pipelines * 8 flops per vector register
PlatformParams perlmutterParams(3.9, //alpha 
                                2406.87, //internode beta
                                4234.33, //intranode beta
                                128, 4, //cores, gpus
                                (3.5*1e9)*2*2*8, //peak FLOPS
                                43478 //memBW
                                );

} //autotuning
}//combblas
#endif
