
#ifndef COMMMODEL_H
#define COMMMODEL_H


#include "common.h"



namespace combblas {
namespace autotuning {


/* Generic model for communication time */
class CommModel {

public:
    CommModel(){}

    virtual double ComputeTime(bool inter) {throw std::runtime_error("This method should never be called");}

    virtual int GetWorld() {throw std::runtime_error("This method should never be called");}

};


/* T = alpha + bytes/beta */
template <typename IT>
class PostCommModel : public CommModel {

public:

    PostCommModel(double alpha, double internodeBeta, double intranodeBeta, 
                    std::function<int()> ComputeNumMsgs, std::function<IT()> ComputeNumBytes):
        alpha(alpha), internodeBeta(internodeBeta), intranodeBeta(intranodeBeta), 
        ComputeNumMsgs(ComputeNumMsgs), ComputeNumBytes(ComputeNumBytes)
    {

    }

    double ComputeTime(bool inter) {
        if (inter)
            return ComputeNumMsgs() * alpha + (ComputeNumBytes())/internodeBeta;
        else
            return ComputeNumMsgs() * alpha + (ComputeNumBytes())/intranodeBeta;
    }

private:
    double alpha; double internodeBeta; double intranodeBeta;
    int coresPerNode;
    std::function<int()> ComputeNumMsgs; std::function<IT()> ComputeNumBytes;

};




};
};

#endif


