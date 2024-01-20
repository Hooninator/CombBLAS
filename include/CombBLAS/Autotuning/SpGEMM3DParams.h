
#ifndef SPGEMM3DPARAMS_H
#define SPGEMM3DPARAMS_H


#include "common.h"
#include "SpGEMM3DMatrixInfo.h"
#include "SymbolicSpParMat3D.h"
#include "CommModel.h"
#include "CompModel.h"
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


    static std::vector<SpGEMM3DParams> ConstructSearchSpace(PlatformParams& params) {
        std::vector<SpGEMM3DParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=params.GetCoresPerNode(); _ppn*=2) {
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

        CommModel<AIT> *bcastModel = new PostCommModel<AIT>(platformParams.GetInternodeAlpha(),
                                                    platformParams.GetInternodeBeta(),
                                                     platformParams.GetIntranodeBeta());
        //MakeBcastModelPost(Ainfo, platformParams, BCAST_TREE);

        double bcastATime = BcastTime(bcastModel, Ainfo, platformParams);
        double bcastBTime = BcastTime(bcastModel, Binfo, platformParams);
        
        CompModel *localMultModel = MakeLocalMultModelReg(Ainfo, Binfo, platformParams);
        double localMultTime = LocalMultTime(localMultModel);

#ifdef PROFILE
        statPtr->Log("BcastA time " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Log("BcastB time " + std::to_string(bcastBTime/1e6) + "s");
        statPtr->Log("LocalMult time " + std::to_string(localMultTime/1e6) + "s");
#endif
        delete bcastModel;
        delete localMultModel;

        return 0;
    }


    /* BROADCAST MODELS */

    enum BcastAlgorithm {
        BCAST_TREE,
        BCAST_RING
    } typedef BcastAlgorithm;

    /* Estimate time for all bcasts */

    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, PlatformParams &params) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif

        int gridSize = (nodes*ppn) / layers;
        
        CommOpts * opts = new CommOpts{
            gridSize <= params.GetCoresPerNode() ? true : false //intranode
        };

        CommInfo<IT> * info = MakeBcastCommInfo(Minfo, params);
        
        double singleBcastTime = bcastModel->ComputeTime(info, opts);

        double finalTime = singleBcastTime * sqrt(gridSize);

        delete info;
        
#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
#endif
        return finalTime;
    }


    template <typename IT, typename NT, typename DER>
    CommInfo<IT> * MakeBcastCommInfo(SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, PlatformParams &params) {
        
        IT localNnzApprox = ApproxLocalNnzDensity(Minfo);
        IT msgSize = localNnzApprox * Minfo.GetNzvalSize() +
                        localNnzApprox * Minfo.GetIndexSize() +
                        (localNnzApprox + 1) * Minfo.GetIndexSize();

        int bcastWorldSize = static_cast<int>(sqrt( (this->nodes*this->ppn) / this->layers ));

        BcastAlgorithm alg = SelectBcastAlg(msgSize);
        
        CommInfo<IT> * info = new CommInfo<IT>();

        switch(alg) {

            case BCAST_TREE:
            {

                info->numMsgs = static_cast<int>(log2(bcastWorldSize));
                info->numBytes = msgSize*static_cast<IT>(log2(bcastWorldSize));

                break;
            }
            
            case BCAST_RING:
            {

                info->numMsgs = static_cast<int>(log2(bcastWorldSize)) + (bcastWorldSize-1);
                info->numBytes = static_cast<IT>(std::lround(msgSize*2*( static_cast<float>(bcastWorldSize - 1) / 
                                                                            static_cast<float>(bcastWorldSize) )));

                break;
            }

#ifdef PROFILE
            statPtr->Log("Local nnz estimate: " + std::to_string(localNnzApprox));
            statPtr->Log("Send bytes estimate: " + std::to_string(info->numBytes));
            statPtr->Log("Num msgs: " + std::to_string(info->numMsgs));
#endif
        }
        
        return info;

    }

    template <typename IT>
    BcastAlgorithm SelectBcastAlg(IT msgSize) {
        /*TODO*/
        return BCAST_TREE;
    }



    /* LOCAL SpGEMM MODELS */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    PeakCompModel * MakeLocalMultModelPeak(SpGEMM3DMatrixInfo<AIT, ANT, ADER>& Ainfo, 
                                            SpGEMM3DMatrixInfo<BIT, BNT, BDER>& Binfo,
                                            PlatformParams& params) {
    
        std::function<long()> _ComputeFLOPS = [&Ainfo, &Binfo, this]() {

            return ApproxLocalMultFLOPSDensity(Ainfo, Binfo);    

        };
        
        return new PeakCompModel(params.GetPeakFLOPS(), _ComputeFLOPS);
    
    } 
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    RegressionCompModel * MakeLocalMultModelReg(SpGEMM3DMatrixInfo<AIT, ANT, ADER>& Ainfo, 
                                            SpGEMM3DMatrixInfo<BIT, BNT, BDER>& Binfo,
                                            PlatformParams& params) {
        
        std::function<long()> _ComputeFLOPS = [&Ainfo, &Binfo, this]() {

            return ApproxLocalMultFLOPSDensity(Ainfo, Binfo);    

        };
        
        return new RegressionCompModel(autotuning::regSpGEMMPerlmutter, _ComputeFLOPS);

    }
    
    /* Estimate time for local multiply */
    //JB: two things to try here, complicated dist hash table based count, more accurate, but simple heuristic maybe enough

    double LocalMultTime(CompModel * model){
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        double finalTime = model->ComputeTime();

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("LocalMult calc time " + std::to_string(t1) + "s");
#endif
        return finalTime;
    }

    double LayerMergeTime(){return 0;}
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;}


    /* Approximation functions  */

    /* Approximate local nnz using matrix density 
     * This actually just computes the avg nnz per processor
     */
    template <typename IT, typename NT, typename DER>
    IT ApproxLocalNnzDensity(SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo) {
        int totalProcs = this->nodes * this->ppn ;

        IT localNcols = Minfo.LocalNcols(totalProcs);
        IT localNrows = Minfo.LocalNrows(totalProcs);
        IT localMatSize = localNcols * localNrows;

        IT localNnzApprox = static_cast<IT>(Minfo.GetDensity() * localMatSize);
        return localNnzApprox ;
    }

    //JB: Could also try actually counting nnz given the initial 2D distribution
    
    
    /* Approximate local FLOPS using density-based nnz estimation */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    long ApproxLocalMultFLOPSDensity(SpGEMM3DMatrixInfo<AIT, ANT, ADER>& Ainfo, SpGEMM3DMatrixInfo<BIT, BNT, BDER>& Binfo){
        
        int totalProcs = this->nodes * this->ppn;
        int gridSize = totalProcs / this->layers;
        
        long tileFLOPS = Ainfo.GetDensity() * (Ainfo.GetNrows() / LongSqrt(gridSize)) * // estimate nnz per col of A
                        Binfo.GetDensity() * (Binfo.GetNrows() / (this->layers * LongSqrt(gridSize))) * // estimate nnz per col of B
                        (Binfo.GetNcols() / LongSqrt(gridSize)); // once per col of B
        long localFLOPS = tileFLOPS * LongSqrt(gridSize); //we do sqrt(gridSize) local multiplies

#ifdef PROFILE
        statPtr->Log("Local FLOPS " + std::to_string(localFLOPS));
#endif

        return localFLOPS; 
 
    }


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    AIT ApproxLocalMultFLOPSSymb(SpGEMM3DMatrixInfo<AIT, ANT, ADER>& Ainfo, SpGEMM3DMatrixInfo<BIT, BNT, BDER>& Binfo){
        
    }


    /* UTILITY FUNCTIONS */

    static bool IsPerfectSquare(int num) {
        int root = static_cast<int>(sqrt(num));
        return root*root==num;
    }

    
    template <typename T>
    inline long LongSqrt(T n) {return static_cast<long>(sqrt(n));}



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





