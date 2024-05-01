


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
        ss<<nodes<<","<<ppn<<","<<layers;
        return ss.str();
    }


    //TODO: Probably some smart way to make this more generic
    static std::vector<SpGEMMParams> ConstructSearchSpace3D(PlatformParams& params) {
        std::vector<SpGEMMParams> result;
        for (int _nodes = 1; _nodes<=jobPtr->nodes; _nodes*=2) {
            for (int _ppn=1; _ppn<=params.GetCoresPerNode(); _ppn*=2) {
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


    static std::vector<SpGEMMParams> ConstructSearchSpace2D(PlatformParams& params, uint32_t nodeLimit, uint32_t ppnLimit) {
        std::vector<SpGEMMParams> space;
        for (uint32_t _nodes = 1; _nodes<=nodeLimit; _nodes*=2) {
            for (uint32_t _ppn=1; _ppn<=ppnLimit; _ppn*=2) {
                if (IsPerfectSquare(_ppn*_nodes) ) {
                    space.push_back(SpGEMMParams(_nodes,_ppn,1));
                }
            }
        }

        return space;

    }

    //TODO: I think we should split this into SpGEMM2DParams and SpGEMM3DParams...
    std::shared_ptr<CommGrid> MakeGridFromParams() 
    {

        std::vector<int> ranks; // Ranks to include in the new communicator
        ranks.reserve(nodes*ppn);
        
        // Get all ranks within this parameter object's nodes and ppn
        for (int rank=0; rank<autotuning::worldSize; rank++)
        {
            int nodeRank = rank / jobPtr->tasksPerNode;
            int localRank = rank % jobPtr->tasksPerNode;
            if (nodeRank < nodes && localRank < ppn)
            {
                ranks.push_back(rank);
            }
        }

        // Make the new group 
        MPI_Group worldGroup;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

        MPI_Group newGroup;
        MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &newGroup);

        MPI_Comm newComm;
        MPI_Comm_create_group(MPI_COMM_WORLD, newGroup, 0, &newComm);

        int newSize;
        MPI_Comm_size(newComm, &newSize);

        if (newComm != MPI_COMM_NULL) {

            ASSERT(newSize==nodes*ppn,
            "Expected communicator size to be " + std::to_string(nodes*ppn) + ", but it was "
            + std::to_string(newSize));

            ASSERT(IsPerfectSquare(newSize / layers),
            "Each 2D grid must be a perfect square, instead got " + OutStr());

            std::shared_ptr<CommGrid> newGrid;
            newGrid.reset(new CommGrid(newComm, 0, 0));

            return newGrid;

        } else {
            return NULL;
        }

    }


    template <typename IT, typename NT, typename DER>
    DER ReDistributeMat(SpMat<IT, NT, DER>& mat)
    {
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

std::ostream& operator<<(std::ostream& os, SpGEMMParams& params) {
    os<<params.OutStr();
    return os;
};

}//autotuning
}//combblas





#endif

