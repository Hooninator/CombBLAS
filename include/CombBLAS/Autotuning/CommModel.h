

#include <functional>
#include <exception>

#include "common.h"


#ifndef COMMMODEL_H
#define COMMMODEL_H

namespace combblas {
namespace autotuning {

/* Generic model for communication time */
class CommModel {

public:
    CommModel(){}

    virtual double ComputeTime() {throw std::runtime_error("This method should never be called");}

    virtual int GetWorld() {throw std::runtime_error("This method should never be called");}

};

/* T = alpha + bytes/beta */
template <typename IT>
class PostCommModel : public CommModel {

public:

    PostCommModel(double alpha, double beta, std::function<int()> ComputeNumMsgs, std::function<IT()> ComputeNumBytes):
    alpha(alpha), beta(beta), ComputeNumMsgs(ComputeNumMsgs), ComputeNumBytes(ComputeNumBytes)
    {

    }

    inline double ComputeTime() {
        return ComputeNumMsgs() * alpha + (ComputeNumBytes())/beta;
    }

private:
    double alpha; double beta;
    std::function<int()> ComputeNumMsgs; std::function<IT()> ComputeNumBytes;

};




};
};

#endif


