
#ifndef SPGEMM3DPARAMS_H
#define SPGEMM3DPARAMS_H


#include "common.h"
#include "SpGEMM3DMatrixInfo.h"
#include "SymbolicSpParMat3D.h"
#include "CommModel.h"
#include "BcastInfo.h"
#include "LocalSpGEMMModel.h"
#include "MergeModel.h"
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
        nodes(nodes), ppn(ppn), layers(layers),
        totalProcs(nodes*ppn)
    {
        gridSize = totalProcs / layers;
        rowSize = RoundedSqrt<int,int>(gridSize);
    }


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
        double bcastATime = BcastTime(bcastModel, Ainfo, platformParams);
        double bcastBTime = BcastTime(bcastModel, Binfo, platformParams);
        
        LocalSpGEMMModel * localMultModel = new RegressionLocalSpGEMMModel(autotuning::regSpGEMMPerlmutter);
        double localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo);

#ifdef PROFILE
        statPtr->Log("BcastA time " + std::to_string(bcastATime/1e6) + "s");
        statPtr->Log("BcastB time " + std::to_string(bcastBTime/1e6) + "s");
        statPtr->Log("LocalMult time " + std::to_string(localMultTime/1e6) + "s");
#endif
        delete bcastModel;
        delete localMultModel;

        return 0;
    }


    /* BROADCAST */

    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpGEMM3DMatrixInfo<IT,NT,DER>& Minfo, PlatformParams &params) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        CommOpts * opts = new CommOpts{
            //gridSize <= params.GetCoresPerNode() ? true : false //intranode
            false
        };
        CommInfo<IT> * info = MakeBcastCommInfo<SpGEMM3DMatrixInfo<IT,NT,DER>, IT>(Minfo, rowSize, totalProcs);

        double singleBcastTime = bcastModel->ComputeTime(info, opts);
        double finalTime = singleBcastTime * sqrt(gridSize);

        delete info;
        delete opts;
        
#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("Bcast calc time " + std::to_string(t1) + "s");
#endif

        return finalTime;
    }


    /* LOCAL SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double LocalMultTime(LocalSpGEMMModel * model, 
                            SpGEMM3DMatrixInfo<AIT,ANT,ADER>& Ainfo,
                            SpGEMM3DMatrixInfo<BIT,BNT,BDER>& Binfo) {
#ifdef PROFILE
        auto stime1 = MPI_Wtime();
#endif
        
        long long localFLOPS = model->ApproxLocalMultFLOPSDensity(Ainfo, Binfo, totalProcs, gridSize);
        LocalSpGEMMInfo * info = new LocalSpGEMMInfo { localFLOPS };

        double finalTime = model->ComputeTime(info);

        delete info;

#ifdef PROFILE
        auto etime1 = MPI_Wtime();
        auto t1 = (etime1-stime1);
        statPtr->Log("LocalMult calc time " + std::to_string(t1) + "s");
#endif

        return finalTime;
    }


    double LayerMergeTime() {
        return 0;
    }
 
    double AlltoAllTime(){return 0;}
    double MergeFiberTime(){return 0;}


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
    int totalProcs;
    int gridSize;
    int rowSize;


};//SpGEMM3DParams



}//autotuning
}//combblas

#endif





