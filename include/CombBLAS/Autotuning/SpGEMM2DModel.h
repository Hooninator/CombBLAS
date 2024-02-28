
#ifndef SPGEMM2DMODEL_H
#define SPGEMM2DMODEL_H


#include "common.h"
#include "SpParMatInfo.h"
#include "CommModel.h"
#include "BcastInfo.h"
#include "LocalSpGEMMModel.h"
#include "MergeModel.h"
#include "SpGEMMParams.h"
#include "PlatformParams.h"


namespace combblas {
namespace autotuning {

template <typename AIT, typename ANT,typename ADER,typename BIT,typename BNT,typename BDER>
class SpGEMM2DInputs {

public:

	SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER>()
	{
	}

};

template <typename MT>
class SpGEMM2DModel {
public:

    SpGEMM2DModel(PlatformParams& platformParams ) : platformParams(platformParams)
    {

    }
    

    /* Get runtime estimate of a certain combo of parameters */
    template <typename I>
    double EstimateRuntime(I& inputs, SpGEMMParams& params) { 
        return static_cast<MT *>(this)->EstimateRuntimeImpl(inputs, params);
    }



protected:

    PlatformParams platformParams;

};


class SpGEMM2DModelAnalytical : public SpGEMM2DModel<SpGEMM2DModelAnalytical> {
public:

    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoAnalytical : public SpParMatInfo<IT,NT,DER> {
    public:
		
		/* (row,col,nnz) */
		//TODO: For col split, no need to store row idx, and for row split, no need to store col idx
		typedef std::vector<std::tuple<IT,IT,IT>> NnzTuples;

		//enum SPLIT {COL_SPLIT, ROW_SPLIT};
		using SpParMatInfo<IT,NT,DER>::SpParMatInfo; 
		using SpParMatInfo<IT,NT,DER>::locNnz; 
		using SpParMatInfo<IT,NT,DER>::locNcolsExact; 
		using SpParMatInfo<IT,NT,DER>::locNrowsExact; 
		using SpParMatInfo<IT,NT,DER>::locNcols; 
		using SpParMatInfo<IT,NT,DER>::locNrows; 
		using SpParMatInfo<IT,NT,DER>::locMat; 
		using SpParMatInfo<IT,NT,DER>::split; 
		using SpParMatInfo<IT,NT,DER>::rowRank; 
		using SpParMatInfo<IT,NT,DER>::colRank; 
		using SpParMatInfo<IT,NT,DER>::gridDims; 
		using SpParMatInfo<IT,NT,DER>::globDensity; 

		
		SpParMatInfoAnalytical(SpParMat<IT,NT,DER>& Mat): 
			SpParMatInfo<IT,NT,DER>(Mat),
			nnzArr(new std::vector<IT>(0)),
			locDensityArr(new std::vector<float>(worldSize))
		{
			
			locDensityArr->insert(locDensityArr->begin() + rank,
									static_cast<float>(locNnz) / static_cast<float>(locNcolsExact*locNrowsExact));
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_FLOAT, (void*)(locDensityArr->data()), 1, MPI_FLOAT, MPI_COMM_WORLD);

			split = COL_SPLIT; // This is much nicer, and in 2d it doesn't matter

#ifdef NNZ_TUPLES_COL

#ifdef PROFILE
			infoPtr->StartTimer("nnzTuplesColInit");
#endif
			nnzTuples = NnzTuplesCol();
#ifdef PROFILE
			infoPtr->EndTimer("nnzTuplesColInit");
#endif

#endif
			MPI_Barrier(MPI_COMM_WORLD);

        }

		/* Create array of tuples containing nnz per tile column for this processor's local tile  */
		NnzTuples * NnzTuplesCol() {

#ifdef PROFILE
			infoPtr->StartTimer("locNnzTuplesColInit");
#endif

			auto _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
			_nnzTuples->reserve(locNcolsExact);

			// Init local data
			int locTupleSize = 0;
			for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
				if (colIter.nnz()>NNZ_THRESH) {
					_nnzTuples->push_back( std::tuple<IT,IT,IT>{colRank,  colIter.colid() + locNcols*rowRank, colIter.nnz()} );
				}
			}

#ifdef PROFILE
			infoPtr->EndTimer("locNnzTuplesColInit");
#endif

#ifdef DEBUG
			debugPtr->Log("locNnzTuples col");
			for (int i=0; i<_nnzTuples->size(); i++) {
				debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
			}
#endif

			return _nnzTuples;

		}



		/* Initialize array of tuples containing nnz per tile row on this processor's local tile */
		NnzTuples * NnzTuplesRow() {

#ifdef PROFILE
			infoPtr->StartTimer("locNnzTuplesRowInit");
#endif

			// JB: I can't figure out a way to avoid mutating nnz during iteration, so we can't just use std::tuple
			std::map<std::tuple<IT,IT>, IT> nnzMap;
			for (auto colIter = locMat->begcol(); colIter != locMat->endcol(); colIter++) {
				for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
					std::tuple<IT,IT> t{nzIter.rowid() + locNrows*colRank, rowRank};
					nnzMap.emplace(t, 0);
					nnzMap[t] += 1;
				}
			}


			auto  _nnzTuples = new std::vector<std::tuple<IT,IT,IT>>;
			_nnzTuples->reserve(locNrowsExact);

			std::for_each(nnzMap.begin(), nnzMap.end(),
				[&_nnzTuples](auto& elem)  {
					std::tuple<IT,IT,IT> t{std::get<0>(elem.first), std::get<1>(elem.first), elem.second};
					_nnzTuples->push_back( t );
				}
			);

#ifdef PROFILE
			infoPtr->EndTimer("locNnzTuplesRowInit");
#endif

#ifdef DEBUG
			debugPtr->Log("locNnzTuples row");
			for (int i=0; i<_nnzTuples->size(); i++) {
				debugPtr->Log(std::to_string(i) + ":" + TupleStr(_nnzTuples->at(i)));
			}
#endif

			return _nnzTuples;

		}


		/* Approximate local nnz using matrix globDensity
		 * This actually just computes the avg nnz per processor
		 */
		IT ComputeLocNnzGlobDensity() {

			IT localNcols = std::get<1>(gridDims);
			IT localNrows = std::get<0>(gridDims);
			IT localMatSize = localNcols * localNrows;

			IT localNnzApprox = static_cast<IT>(globDensity * localMatSize);
			return localNnzApprox;
		}


		/* Approximate local nnz using matrix locDensityArr
		 */
		IT ComputeLocNnzLocDensity(int procRank) {

			IT localNcols = std::get<1>(gridDims);
			IT localNrows = std::get<0>(gridDims);
			IT localMatSize = localNcols * localNrows;

			IT localNnzApprox = static_cast<IT>(locDensityArr->at(procRank) * localMatSize);
			return localNnzApprox;
		}


		void ComputeNnzArr(SpGEMMParams& params) {

#ifdef PROFILE
			infoPtr->StartTimer("ComputeNnzArr");
#endif

			nnzArr->clear();
			nnzArr->resize(params.GetTotalProcs());

			switch(split) {
				case COL_SPLIT:
				{
					ComputeNnzArrColSplit(params);
					break;
				}
				case ROW_SPLIT:
				{
					ComputeNnzArrRowSplit(params);
					break;
				}
				default:
				{
					UNREACH_ERR();
				}
			}

#ifdef PROFILE
			infoPtr->EndTimer("ComputeNnzArr");
#endif

		}

		
		/* Given local nnz in initial 2D processor grid, compute nnz per processor in 3D processr grid
		 * WITHOUT explicitly forming the 3D processor grid. */
		void ComputeNnzArrColSplit(SpGEMMParams& params) {

			const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_COL
			// Local nnz array
			std::for_each(nnzTuples->begin(), nnzTuples->end(),
				[&params,this](auto& t) {
					int i = std::get<0>(t);
					int j = std::get<1>(t);
					int owner = ComputeOwnerGrid(params, i*this->locNrows, j, COL_SPLIT);
					this->nnzArr->at(owner) += std::get<2>(t);
				}
			);
#else
			// Just use local matrix
			for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
				int j = colIter.colid();
				for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
					int i = nzIter.rowid();
					int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), COL_SPLIT);
					nnzArr->at(owner) += 1;
				}
			}
#endif

			// Allreduce to get complete counts for each process
			MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
		 debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr A: "});
#endif

		}
		


		void ComputeNnzArrRowSplit(SpGEMMParams& params) {

			const int totalProcs = params.GetTotalProcs();

#ifdef NNZ_TUPLES_ROW
			// Local data
			std::for_each(nnzTuples->begin(), nnzTuples->end(),
				[&params, this](auto& t) {
					int i = std::get<0>(t);
					int j = std::get<1>(t);
					int owner = ComputeOwnerGrid(params, i, j*this->locNcols, ROW_SPLIT);
					this->nnzArr->at(owner) += std::get<2>(t);
				}
			);
#else
			for (auto colIter = locMat->begcol(); colIter!=locMat->endcol(); colIter++) {
				int j = colIter.colid();
				for (auto nzIter = locMat->begnz(colIter); nzIter!=locMat->endnz(colIter); nzIter++) {
					int i = nzIter.rowid();
					int owner = ComputeOwnerGrid(params, i+(colRank*locNrows), j+(rowRank*locNcols), ROW_SPLIT);
					nnzArr->at(owner) += 1;
				}
			}
#endif

			// Allreduce to sum all nnz
			MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzArr->data()), totalProcs, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);

#ifdef DEBUG
			debugPtr->LogVecSameLine(*nnzArr, std::string{"nnzArr B: "});
#endif

		}
		
		IT ComputeLocNnzGrid(NNZ_STRAT strat, int procRank) {
			switch(strat) {
				case NNZ_GLOB_DENSITY:
					return ComputeLocNnzGlobDensity();
				case NNZ_LOC_DENSITY:
					return ComputeLocNnzLocDensity(procRank);
				case NNZ_ARR:
					return nnzArr->at(procRank);
				default:
					UNREACH_ERR();
			}
			return 0;
		}

	
		int ComputeOwnerGrid(SpGEMMParams& params, const int i, const int j, SPLIT split) {

			const int layers = params.GetLayers();
			const int gridDim = params.GetGridDim();
			const int gridSize = params.GetGridSize();

			IT locNrowsGrid = std::get<0>(gridDims);
			IT locNcolsGrid = std::get<1>(gridDims);

			IT colDiv;
			IT rowDiv;
			IT layerDiv;

			int layerIdx;

			if (split==COL_SPLIT) {
				colDiv = locNcolsGrid*layers;
				rowDiv = locNrowsGrid;
				layerDiv = locNcolsGrid;
				layerIdx = j;
			} else if (split==ROW_SPLIT) {
				colDiv = locNcolsGrid;
				rowDiv = locNrowsGrid*layers;
				layerDiv = locNrowsGrid;
				layerIdx = i;
			}

			const int prow = std::min(static_cast<IT>(i / rowDiv), static_cast<IT>(gridDim-1));
			const int pcol = std::min(static_cast<IT>(j / colDiv), static_cast<IT>(gridDim-1));
			const int player = std::min(static_cast<IT>((layerIdx / layerDiv)%layers), static_cast<IT>(layers-1));

			return (pcol + prow*gridDim + player*gridSize);
		}
	
		
		/* Sum nnz in procRank's row of the hypothetical 3D grid */
		std::vector<IT> SliceNnzRow(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
			return std::vector<IT>(nnzArr->begin()+(procRank/gridDim), nnzArr->begin()+(procRank/gridDim)+gridDim);
		}


		/* Sum nnz in procRank's column of hypothetical 3D grid */
		std::vector<IT> SliceNnzCol(const std::vector<IT> * nnzArr, const int procRank, const int gridDim) {
			//TODO: Can we use C++17 algorithms for this?
			std::vector<IT> result(gridDim);
			for (int p=0; p<gridDim; p++) {
				result[p] = nnzArr->at((procRank%gridDim)+p*gridDim);
			}
			return result;
		}


		inline std::vector<IT> * GetNnzArr() {return nnzArr;}
		inline std::vector<float> * GetLocDensityArr() const {return locDensityArr;}
		
	private:

		std::vector<float> * locDensityArr;
		NnzTuples * nnzTuples;

		// Stores nnz per processor in hypothetical 3D grid
		std::vector<IT> * nnzArr;
		

    };


    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {

    public:

        Inputs<AIT,ANT,ADER,BIT,BNT,BDER>(SpParMat<AIT,ANT,ADER>& A,
                                                    SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A),Binfo(B)
        {
        }

        SpParMatInfoAnalytical<AIT,ANT,ADER> Ainfo;
        SpParMatInfoAnalytical<BIT,BNT,BDER> Binfo;
    };


    /* Get runtime estimate of a certain combo of parameters */
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntimeImpl(Inputs<AIT,ANT,ADER, BIT, BNT, BDER>& inputs, SpGEMMParams& params) {
        
        //Inputs<AIT,ANT,ADER,BIT,BNT,BDER> inputs = static_cast<Inputs<AIT,ANT,ADER,BIT,BNT,BDER>>(baseInputs);

#ifdef DEBUG
        debugPtr->Log(params.OutStr());
        debugPtr->Print0(params.OutStr());
#endif

#ifdef PROFILE
        infoPtr->Put("Nodes", std::to_string(params.GetNodes()));
        infoPtr->Put("PPN", std::to_string(params.GetPPN()));
        infoPtr->Print("Nodes");
        infoPtr->Print("PPN");
#endif

        auto Ainfo = inputs.Ainfo;
        auto Binfo = inputs.Binfo;

        // Set dimensions of 3D processor grid
        Ainfo.SetGridDims(params);
        Binfo.SetGridDims(params);

        // Compute nnz per tile in hypothetical 3D grid
        Ainfo.ComputeNnzArr(params);
        Binfo.ComputeNnzArr(params);

        //BROADCAST
        CommModel<AIT> *bcastModel = new PostCommModel<AIT>(platformParams.GetInternodeAlpha(),
                                                    platformParams.GetInternodeBeta(),
                                                     platformParams.GetIntranodeBeta());
        double bcastATime = BcastTime(bcastModel, Ainfo, params, true);
        double bcastBTime = BcastTime(bcastModel, Binfo, params, false);
        
        //LOCAL SpGEMM
        LocalSpGEMMModel<AIT, BIT>* localMultModel = new RooflineLocalSpGEMMModel<AIT, ANT, BIT, BNT>(autotuning::perlmutterParams);
        double localMultTime = LocalMultTime(localMultModel, Ainfo, Binfo, params);

#ifdef PROFILE
        infoPtr->Put("bcastTime-A", std::to_string(bcastATime/1e6));
        infoPtr->Put("bcastTime-B", std::to_string(bcastBTime/1e6));
        infoPtr->Put("multTime", std::to_string(localMultTime/1e6));
#endif

        delete bcastModel;
        delete localMultModel;

        MPI_Barrier(MPI_COMM_WORLD);

        return bcastATime + bcastBTime + localMultTime;
    }


    /* BROADCAST */

    //TODO: Consider nnz estimator class + template to make switching between things here easier
    template <typename IT, typename NT, typename DER>
    double BcastTime(CommModel<IT> * bcastModel, SpParMatInfoAnalytical<IT,NT,DER>& Minfo, SpGEMMParams& params, bool row) {

#ifdef PROFILE
        if (row)
            infoPtr->StartTimer("bcastCalcTime-A");
        else
            infoPtr->StartTimer("bcastCalcTime-B");
#endif

        std::vector<IT> * nnz2D = Minfo.GetNnzArr();

        // Compute local bcast times
        std::vector<double> locBcastTimes(params.GetTotalProcs());
        for (int p=0; p<params.GetTotalProcs(); p++) {
            
            // Vector containing nnz for each rank participating in broadcasts with rank p
            std::vector<IT> nnzBcastWorld(params.GetGridDim());
            //TODO: Params class should have methods that return ranks in row/col, then just use std::transform to create bcast world
            if (row) 
                nnzBcastWorld = Minfo.SliceNnzRow(nnz2D, p, params.GetGridDim());
            else
                nnzBcastWorld = Minfo.SliceNnzCol(nnz2D, p, params.GetGridDim());
            
            // Compute and sum all times for all bcasts rank p participates in 
            double locBcastTime = std::reduce(nnzBcastWorld.begin(), nnzBcastWorld.end(), 0, 
                [&Minfo, &bcastModel, &params](double sum, IT nnz) {
                    IT msgSize = Minfo.ComputeMsgSize(nnz);

                    CommOpts * opts = new CommOpts{
                        //gridSize <= params.GetCoresPerNode() ? true : false //intranode
                        false
                    };

                    CommInfo<IT> * info = MakeBcastCommInfo(params.GetGridDim(),  msgSize); 

                    double singleBcastTime = bcastModel->Time(info, opts);

                    delete info;
                    delete opts;

                    return singleBcastTime + sum;
                }
            );
            
            locBcastTimes[p] = locBcastTime;

        }

        // Reduce to get max time
        double finalTime = std::reduce(locBcastTimes.begin(), locBcastTimes.end(), 0,
            [](double currMax, double currElem) {
                return std::max(currMax, currElem);
            }
        );

#ifdef PROFILE
        if (row) {
            infoPtr->EndTimer("bcastCalcTime-A");
            infoPtr->Print("bcastCalcTime-A");
        } else {
            infoPtr->EndTimer("bcastCalcTime-B");
            infoPtr->Print("bcastCalcTime-B");
        }
#endif

        return finalTime;
    }


    /* LOCAL SpGEMM */
    
    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double LocalMultTime(LocalSpGEMMModel<AIT, BIT>* model, 
                            SpParMatInfoAnalytical<AIT,ANT,ADER>& Ainfo,
                            SpParMatInfoAnalytical<BIT,BNT,BDER>& Binfo,
                            SpGEMMParams& params) {
#ifdef PROFILE
        infoPtr->StartTimer("multCalcTime");
#endif
        
        auto Adims = Ainfo.GetGridDims(); 
        auto Bdims = Binfo.GetGridDims();

        const int totalProcs = params.GetTotalProcs();

        std::vector<double> * localSpGEMMTimes = new std::vector<double>;
        localSpGEMMTimes->reserve(totalProcs);
        for (int p=0; p<totalProcs; p++) {

            auto ranksA = Ainfo.RowRanks(p, params);
            auto ranksB = Binfo.ColRanks(p, params);

            ASSERT(ranksA.size()==ranksB.size(), "ranksA and ranksB should be the same size, instead got " +
                                            std::to_string(ranksA.size()) +  "," + std::to_string(ranksB.size()));

            for (int i=0; i<ranksA.size(); i++) {
                int rankA = ranksA[i];
                int rankB = ranksB[i];
                LocalSpGEMMInfo<AIT, BIT> * info = new LocalSpGEMMInfo<AIT, BIT> 
                                                    { -1, //placeholder 
                                                    std::get<0>(Adims), std::get<1>(Adims),
                                                    std::get<0>(Bdims), std::get<1>(Bdims),
                                                    Ainfo.ComputeLocNnzGrid(NNZ_ARR,rankA), 
                                                    Binfo.ComputeLocNnzGrid(NNZ_ARR,rankB),
                                                    Ainfo.GetGlobDensity(),
                                                    Ainfo.GetLocDensityArr()->at(rankA),
                                                    Binfo.GetGlobDensity(),
                                                    Binfo.GetLocDensityArr()->at(rankB)};
                info->SetFLOPS(params, FLOPS_LOC_DENSITY);
                localSpGEMMTimes->push_back(model->Time(info));
            }

        }


        // Reduce to get max time
        double finalTime = std::reduce(localSpGEMMTimes->begin(),localSpGEMMTimes->end(), 0,
            [](double currMax, double currElem) {
                return std::max(currMax, currElem);
            }
        );

        delete localSpGEMMTimes;

#ifdef PROFILE
        infoPtr->EndTimer("multCalcTime");
        infoPtr->Print("multCalcTime");
#endif

        return finalTime;
    }

    double LayerMergeTime() {
        return 0;
    }
 
};


#ifdef XGB_MODEL

class SpGEMM2DModelXgb : public SpGEMM2DModel<SpGEMM2DModelXgb> {
public:
    
    template <typename IT, typename NT, typename DER>
    class SpParMatInfoXgb : public SpParMatInfo<IT,NT,DER> {
    public:
        SpParMatInfoXgb(SpParMat<IT,NT,DER>& Mat):
            SpParMatInfo<IT,NT,DER>(Mat)
        {

        }


		// NOTE: need overloaded function here because behavior differs depending on 2d vs 3d
		void SetGlobalColInfo(SpParMat<IT,NT,DER>& Mat) {

			// avg nnz per column
			avgNnzCol = static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getncol());

			// avg density per column
			avgDensityCol = (static_cast<float>(Mat.getnnz()) / static_cast<float>(Mat.getnrow())) / static_cast<float>(Mat.getncol());

			// Reduce to get complete nnz per column
			std::vector<IT> nnzColVec(Mat.seqptr()->getncol());
			float sumNnzMeanDiff;

			for (auto colIter = Mat.seqptr()->begcol(); colIter!=Mat.seqptr()->endcol(); colIter++) {
				nnzColVec[colIter.colid()] = colIter.nnz();
				sumNnzMeanDiff += std::pow( (colIter.nnz() - avgNnzCol), 2);
			}

			MPI_Allreduce(MPI_IN_PLACE, (void*)(nnzColVec.data()), nnzColVec.size(), MPIType<IT>(), MPI_SUM,
						Mat.getcommgrid()->GetColWorld());

			// Compute column densities
			std::vector<float> densityColVec(Mat.seqptr()->getncol());
			float sumDensityMeanDiff;

			std::transform(nnzColVec.begin(), nnzColVec.end(), densityColVec.begin(),
					[this, &sumDensityMeanDiff](IT nnz) mutable {
						float d = static_cast<float>(nnz) / static_cast<float>(this->nrows);
						sumDensityMeanDiff += std::pow( (d - this->avgDensityCol), 2);
						return d;
					}
			);

			// Local reduce to get min, max and sum for each column block
			float locMinDensity, locMaxDensity;
			minNnzCol = ReduceMin(nnzColVec);
			maxNnzCol = ReduceMax(nnzColVec);
			minDensityCol = ReduceMin(densityColVec);
			maxDensityCol = ReduceMax(densityColVec);

			// Global reduce to compute final min, max, and sum
			// TODO: use nonblocking collectives?
			MPI_Allreduce(MPI_IN_PLACE, (void*)(&minNnzCol), 1, MPIType<IT>(), MPI_MIN, Mat.getcommgrid()->GetRowWorld());
			MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxNnzCol), 1, MPIType<IT>(), MPI_MAX, Mat.getcommgrid()->GetRowWorld());

			MPI_Allreduce(MPI_IN_PLACE, (void*)(&minDensityCol), 1, MPI_FLOAT, MPI_MIN, Mat.getcommgrid()->GetRowWorld());
			MPI_Allreduce(MPI_IN_PLACE, (void*)(&maxDensityCol), 1, MPI_FLOAT, MPI_MAX, Mat.getcommgrid()->GetRowWorld());

			// pack floats that will be summed into single buffer
			float locBuf[] = {sumNnzMeanDiff, sumDensityMeanDiff};
			MPI_Allreduce(MPI_IN_PLACE, (void*)(locBuf), 2, MPI_FLOAT, MPI_SUM, Mat.getcommgrid()->GetRowWorld());

			// finish stdev calculations
			stdevNnzCol = std::sqrt( sumNnzMeanDiff / Mat.getncol() );
			stdevDensityCol = std::sqrt( sumDensityMeanDiff / Mat.getncol() );

		}


		inline IT GetAvgNnzCol() const {return avgNnzCol;}
		inline IT GetMinNnzCol() const {return minNnzCol;}
		inline IT GetMaxNnzCol() const {return maxNnzCol;}
		inline IT GetStdevNnzCol() const {return stdevNnzCol;}

		inline IT GetAvgDensityCol() const {return avgDensityCol;}
		inline IT GetMinDensityCol() const {return minDensityCol;}
		inline IT GetMaxDensityCol() const {return maxDensityCol;}
		inline IT GetStdevDensityCol() const {return stdevDensityCol;}

    private:

		float avgNnzCol;
		IT minNnzCol;
		IT maxNnzCol;
		float stdevNnzCol;
		float avgDensityCol;
		float minDensityCol;
		float maxDensityCol;
		float stdevDensityCol;

    };

    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    class Inputs : public SpGEMM2DInputs<AIT,ANT,ADER,BIT,BNT,BDER> {
    public:
        Inputs(SpParMat<AIT,ANT,ADER>& A, SpParMat<BIT,BNT,BDER>& B):
            Ainfo(A),Binfo(B)
        {
        }

        SpParMatInfoXgb<AIT,ANT,ADER> Ainfo;
        SpParMatInfoXgb<BIT,BNT,BDER> Binfo;
        
    };

    template <typename AIT, typename ANT, typename ADER, typename BIT, typename BNT, typename BDER>
    double EstimateRuntimeImpl(Inputs<AIT,ANT,ADER,BIT,BNT,BDER>& inputs, SpGEMMParams& params) {
#ifdef DEBUG
        debugPtr->Log(params.OutStr());
        debugPtr->Print0(params.OutStr());
#endif

#ifdef PROFILE
        infoPtr->Put("Nodes", std::to_string(params.GetNodes()));
        infoPtr->Put("PPN", std::to_string(params.GetPPN()));
        infoPtr->Print("Nodes");
        infoPtr->Print("PPN");
#endif

        return 0;
    }
};

#endif


}//autotuning
}//combblas

#endif





