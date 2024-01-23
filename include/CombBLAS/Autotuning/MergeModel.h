

#ifndef MERGE_MODEL_H
#define MERGE_MODEL_H

#include "common.h"

namespace combblas {
namespace autotuning {

struct MergeModelParams {

    //TODO

};

class MergeModel {

public:
    virtual double ComputeTime(MergeModelParams * params) {
        throw std::runtime_error("This should never be called");
    }

};


/* Use Sampled Compression Ratio to estimate nnz in output tile */
class MergeModelCompression : public MergeModel {



};


}//autotuning
}//combblas




#endif
