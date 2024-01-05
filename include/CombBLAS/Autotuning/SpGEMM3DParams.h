
#ifndef SPGEMM3DPARAMS_H
#define SPGEMM3DPARAMS_H


#include "common.h"
#include "SpGEMM3DMatrixInfo.h"
#include "CommModel.h"
#include "PlatformParams.h"


namespace combblas {
namespace autotuning {


template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
class SpGEMM3DInputs {

public:

    SpGEMM3DInputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo,
                                                SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo):
    Ainfo(Ainfo), Binfo(Binfo) {}

    SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo;
    SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo;
};


/* Output of tune routine. Relevant SpGEMM3D params that should be used to create a CommGrid3D 
 * Also defines runtime estimation model. TODO: Move that functionality to a dedicated model class
 */
class SpGEMM3DParams {
public:

    SpGEMM3DParams(){}


    SpGEMM3DParams(int nodes, int ppn, int layers):
    nodes(nodes), ppn(ppn), layers(layers) {}


    void Print() {
        std::cout<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")"<<std::endl;
    }


    std::string OutStr() {
        std::stringstream ss;
        ss<< "(Nodes: "<<nodes<<", PPN: "<<ppn<<", Layers: "<<layers<<")";
        return ss.str();
    }


    static std::vector<SpGEMM3DParams> ConstructSearchSpace() {
        std::vector<SpGEMM3DParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=jobPtr->tasksPerNode; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes)) {
                    for (int _layers=1; _layers<=_ppn*_nodes; _layers*=2) {
                        int gridSize = (_ppn*_nodes) / _layers;
                        if (IsPerfectSquare(gridSize))
                            result.push_back(SpGEMM3DParams(_nodes,_ppn,_layers));
                    }
                }
            }
        }
        return result;
    }


    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntime(SpGEMM3DInputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, PlatformParams& platformParams) {

#ifdef DEBUG
        debugPtr->Log(OutStr());
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        CommModel *bcastModelA = MakeBcastModelPost(Ainfo, platformParams, BCAST_TREE);
        CommModel *bcastModelB = MakeBcastModelPost(Binfo, platformParams, BCAST_TREE);

        double bcastATime = BcastTime(bcastModelA);
        double bcastBTime = BcastTime(bcastModelB);

#ifdef PROFILE
        statPtr->Print("[BcastA time] " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Print("[BcastB time] " + std::to_string(bcastBTime/1e6) + "s");
        statPtr->Log("BcastA time " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Log("BcastB time " + std::to_string(bcastBTime/1e6) + "s");
#endif
        delete bcastModelA; delete bcastModelB;

        return 0;
    }


    /* BROADCAST MODELS */

    enum BcastAlgorithm {
        BCAST_TREE,
        BCAST_RING
    } typedef BcastAlgorithm;

    enum BcastWorld {
        BCAST_ROW,
        BCAST_COL
    } typedef BcastWorld;


    template <typename IT, typename NT, typename DER>
    PostCommModel<IT> * MakeBcastModelPost(SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, PlatformParams &params, 
                                            BcastAlgorithm alg) {

        std::function<int()> _ComputeNumMsgs;
        std::function<IT()> _ComputeNumBytes;

        switch(alg) {

            case BCAST_TREE:
            {
                _ComputeNumMsgs = [this]() {
                    int bcastWorldSize = static_cast<int>(sqrt( (this->nodes*this->ppn) / this->layers ));
                    return static_cast<int>(log2(bcastWorldSize));
                };

                _ComputeNumBytes = [&Minfo, this]() {

                    IT localNnzApprox = ApproxLocalNnzDensity(Minfo);

                    IT sendBytes = localNnzApprox * Minfo.GetNzvalSize() +
                                    localNnzApprox * Minfo.GetIndexSize() +
                                    (localNnzApprox + 1) * Minfo.GetIndexSize();

                    int bcastWorldSize = static_cast<int>(sqrt(this->nodes*this->ppn / this->layers));
                    IT totalBytes = sendBytes*static_cast<IT>(log2(bcastWorldSize));
#ifdef PROFILE
                    statPtr->Log("Local nnz estimate: " + std::to_string(localNnzApprox));
                    statPtr->Log("Send bytes estimate: " + std::to_string(totalBytes));
#endif
                    return totalBytes;
                };

                break;
            }

        }

        return new PostCommModel<IT>(params.GetInternodeAlpha(), params.GetInternodeBeta(),
                                    _ComputeNumMsgs, _ComputeNumBytes);
    }


    double BcastTime(CommModel * bcastModel) {

#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif

        double singleBcastTime = bcastModel->ComputeTime();

        int gridSize = (nodes*ppn) / layers;
        double finalTime = singleBcastTime * sqrt(gridSize);

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Print("[Bcast calc time] " + std::to_string(t1) + "s");
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
#endif
        return finalTime;
    }



    double LocalMultTime(){return 0;}
    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;}


    /* Approximation functions  */

    /* Approximate local nnz using matrix density */
    template <typename IT, typename NT, typename DER>
    IT ApproxLocalNnzDensity(SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo) {
        int totalProcs = this->nodes * this->ppn;

        IT localNcols = Minfo.GetNcols() / static_cast<IT>(sqrt(totalProcs));
        IT localNrows = Minfo.GetNrows() / static_cast<IT>(sqrt(totalProcs));
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(Minfo.GetDensity() * localMatSize);
        return localNnzApprox;
    }



    /* UTILITY FUNCTIONS */

    static bool IsPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root*root==num;
    }



    /* Given a set of parameters, construct a 3D processor grid from a communicator that only contains the processes
     * with local ranks < ppn on nodes < n
     */
    std::shared_ptr<CommGrid3D> MakeGridFromParams() {
        //TODO: This needs some major work
        int nodeRank = (autotuning::rank / jobPtr->tasksPerNode);
        int color = static_cast<int>(nodeRank < nodes && autotuning::localRank < ppn);
        int key = autotuning::rank;

        MPI_Comm newComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &newComm);

        int newSize;
        MPI_Comm_size(newComm, &newSize);

        if (color==1) {

            ASSERT(newSize==nodes*ppn,
            "Expected communicator size to be " + std::to_string(nodes*ppn) + ", but it was "
            + std::to_string(newSize));

            ASSERT(IsPerfectSquare(newSize / layers),
            "Each 2D grid must be a perfect square, instead got " + OutStr());

            std::shared_ptr<CommGrid3D> newGrid;
            newGrid.reset(new CommGrid3D(newComm, layers, 0, 0));

            return newGrid;

        } else {
            return NULL;
        }

    }

    /* Tunable parameters */

    int nodes;
    int ppn;
    int layers;


};//SpGEMM3DParams



}//autotuning
}//combblas

#endif





