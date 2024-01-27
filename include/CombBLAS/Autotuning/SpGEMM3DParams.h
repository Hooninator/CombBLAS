


#ifndef SPGEMM3DPARAMS_H
#define SPGEMM3DPARAMS_H

#include "common.h"
#include "PlatformParams.h"


namespace combblas{

namespace autotuning{

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


    //TODO: Probably some smart way to make this more generic
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

    inline int GetNodes() const {return nodes;}
    inline int GetPPN() const {return ppn;}
    inline int GetLayers() const {return layers;}
    inline int GetTotalProcs() const {return totalProcs;}
    inline int GetGridSize() const {return gridSize;}
    inline int GetRowSize() const {return rowSize;}

private:
    /* Tunable parameters */
    int nodes;
    int ppn;
    int layers;

    /* Other handy info */
    int totalProcs;
    int gridSize;
    int rowSize;



};


}//autotuning
}//combblas





#endif

