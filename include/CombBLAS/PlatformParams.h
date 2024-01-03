


#include <exception>


namespace combblas {

namespace autotuning {
/* 
 * TODO: Topology specific params
 * TODO: intrasocket,intranode alpha/beta
 */

class PlatformParams {
public:
    
    PlatformParams() {
        internodeAlpha = MeasureInternodeAlpha();
        internodeBeta = MeasureInternodeBeta();
    }
    
    //alpha is us
    //beta is bytes/us
    PlatformParams(float internodeAlpha, float internodeBeta, int coresPerNode, int devsPerNode) : 
    internodeAlpha(internodeAlpha), internodeBeta(internodeBeta), coresPerNode(coresPerNode), devsPerNode(devsPerNode)
    {}
    ~PlatformParams(){}
    
    float GetInternodeAlpha() const {return internodeAlpha;} 
    float GetInternodeBeta() const {return internodeBeta;} 
    int GetCoresPerNode() const {return coresPerNode;}
    int GetDevsPerNode() const {return devsPerNode;}
    
    float MeasureInternodeAlpha() {throw std::runtime_error("Not implemented");}
    float MeasureInternodeBeta() {throw std::runtime_error("Not implemented");}
    

private:
    
    float internodeAlpha;
    float internodeBeta;
    int coresPerNode;
    int devsPerNode;
    
};

//Values obtained with osu microbenchmarks
PlatformParams perlmutterParams(3.9, 2406.87, 128, 4);

} //autotuning
}//combblas
