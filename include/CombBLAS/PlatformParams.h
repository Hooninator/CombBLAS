


#include <exception>


namespace combblas {

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

    PlatformParams(float internodeAlpha, float internodeBeta) : internodeAlpha(internodeAlpha), internodeBeta(internodeBeta){}
    ~PlatformParams(){}
    
    float GetInternodeAlpha() const {return internodeAlpha;} 
    float GetInternodeBeta() const {return internodeBeta;} 
    
    float MeasureInternodeAlpha() {throw std::runtime_error("Not implemented");}
    float MeasureInternodeBeta() {throw std::runtime_error("Not implemented");}
    

private:
    
    float internodeAlpha;
    float internodeBeta;
    
};

}//combblas
