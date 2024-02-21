


#ifndef SPGEMMPARAMS_H
#define SPGEMMPARAMS_H

#include "common.h"
#include "PlatformParams.h"


namespace combblas{

namespace autotuning{

class SpGEMMParams {
    
public:


	
    SpGEMMParams(){}


    SpGEMMParams(int nodes, int ppn, int layers):
        nodes(nodes), ppn(ppn), layers(layers),
        totalProcs(nodes*ppn)
    {
        gridSize = totalProcs / layers;
        gridDim = RoundedSqrt<int,int>(gridSize);
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
    static std::vector<SpGEMMParams> ConstructSearchSpace3D(PlatformParams& params) {
        std::vector<SpGEMMParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=jobPtr->tasksPerNode; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes)) {
                    for (int _layers=1; _layers<=_ppn*_nodes; _layers*=2) {
                        int gridSize = (_ppn*_nodes) / _layers;
                        if (IsPerfectSquare(gridSize))
                            result.push_back(SpGEMMParams(_nodes,_ppn,_layers));
                    }
                }
            }
        }
        return result;
    }


    static std::vector<SpGEMMParams> ConstructSearchSpace2D(PlatformParams& params) {
        std::vector<SpGEMMParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=jobPtr->tasksPerNode; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes)) {
                    result.push_back(SpGEMMParams(_nodes,_ppn,1));
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

    
    /* Utility functions for getting MPI Communicators across symbolic 3D grid */

    MPI_Comm GridComm() {
        int color = rank / gridSize;
        int key = rank % gridSize;
        MPI_Comm gridComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &gridComm);
        return gridComm;
    }


    MPI_Comm RowComm(MPI_Comm gridComm) {
        int gridRank;
        MPI_Comm_rank(gridComm, &gridRank);
        int color = gridRank / gridDim;
        int key = gridRank % gridDim;
        MPI_Comm rowComm;
        MPI_Comm_split(gridComm, color, key, &rowComm);
        return rowComm;
    }


    MPI_Comm ColComm(MPI_Comm gridComm) {
        int gridRank;
        MPI_Comm_rank(gridComm, &gridRank);
        int color = gridRank % gridDim;
        int key = gridRank / gridDim;
        MPI_Comm colComm;
        MPI_Comm_split(gridComm, color, key, &colComm);
        return colComm;
    }

    MPI_Comm WorldComm() {
        int color = (rank < totalProcs) ? 0 : 1;
        int key = rank;
        MPI_Comm worldComm;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &worldComm);
        return worldComm;
    }


    inline int GetNodes() const {return nodes;}
    inline int GetPPN() const {return ppn;}
    inline int GetLayers() const {return layers;}
    inline int GetTotalProcs() const {return totalProcs;}
    inline int GetGridSize() const {return gridSize;}
    inline int GetGridDim() const {return gridDim;}

private:
    /* Tunable parameters */
    int nodes;
    int ppn;
    int layers;

    /* Other handy info */
    int totalProcs;
    int gridSize;
    int gridDim;



};


}//autotuning
}//combblas





#endif

