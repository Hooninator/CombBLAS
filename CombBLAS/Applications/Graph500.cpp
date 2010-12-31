#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#ifdef NOTR1
        #include <boost/tr1/memory.hpp>
#else
        #include <tr1/memory>
#endif
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../FullyDistVec.h"
#include "../FullyDistSpVec.h"
#include "../ParFriends.h"
#include "../DistEdgeList.h"


using namespace std;

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	MPI::COMM_WORLD.Set_errhandler ( MPI::ERRORS_THROW_EXCEPTIONS );
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./Graph500 <Auto,Force,Input> <Available RAM in MB (per core) | Scale Forced | Input Name>" << endl;
			cout << "Example: ./Graph500 Auto 1024" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}		
	{
		typedef SelectMaxSRing<bool, int64_t> SR;	
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

		// Declare objects
		PSpMat_Bool A;	
		FullyDistVec<int64_t, int64_t> degrees;
		unsigned scale;

		if(string(argv[1]) == string("Input")) // input option
		{
			ifstream input(argv[2]);
			A.ReadDistribute(input, 0);	// read it from file
			SpParHelper::Print("Read input");

			PSpMat_Int64 * G = new PSpMat_Int64(A); 
			G->Reduce(degrees, Row, plus<int64_t>(), 0);	// identity is 0 
			delete G;

			PSpMat_Bool AT = A;
			AT.Transpose();
			// boolean addition is practically a "logical or", 
			// therefore this doesn't destruct any links
			A += AT;	// symmetricize
		}
		else 
		{	
			if(string(argv[1]) == string("Auto"))	
			{
				// calculate the problem size that can be solved
				// number of nonzero columns are at most the matrix dimension (for small p)
				// for large p, though, nzc = nnz since each subcolumn will have a single nonzero 
				// so assume (1+8+8+8)*nedges for the uint64 case and (1+4+4+4)*nedges for uint32
				uint64_t raminbytes = static_cast<uint64_t>(atoi(argv[2])) * 1024 * 1024;	
				uint64_t peredge = 1+3*sizeof(int64_t);
				uint64_t maxnedges = raminbytes / peredge;
				uint64_t maxvertices = maxnedges / 32;	
				unsigned maxscale = highestbitset(maxvertices * nprocs);

				string name;
				if(maxscale > 36)	// at least 37 so it fits comfortably along with vectors 
				{
					name = "Medium";	
					scale = 36;
				}
				else if(maxscale > 32)
				{
					name = "Small";
					scale = 32;
				}
				else if(maxscale > 29)
				{
					name = "Mini";
					scale = 29;
				}
				else if(maxscale > 26)
				{
					name = "Toy";
					scale = 26;
				}
				else
				{
					name = "Debug";
					scale = 20;	// fits even to single processor
				}

				ostringstream outs;
				outs << "Max scale allowed : " << maxscale << endl;
				outs << "Using the " << name << " problem" << endl;
				SpParHelper::Print(outs.str());
			}
			else if(string(argv[1]) == string("Force"))	
			{
				scale = static_cast<unsigned>(atoi(argv[2]));
				ostringstream outs;
				outs << "Forcing scale to : " << scale << endl;
				SpParHelper::Print(outs.str());
			}
			else
			{
				SpParHelper::Print("Unknown option\n");
				MPI::Finalize(); 
				return -1;	
			}
			// this is an undirected graph, so A*x does indeed BFS
 			double initiator[4] = {.57, .19, .19, .05};

			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
			DEL->GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
			SpParHelper::Print("Generated local RMAT matrices\n");
		
			PermEdges<int64_t>(*DEL);
			SpParHelper::Print("Permuted Edges\n");

			RenameVertices<int64_t>(*DEL);	// intermediate: generates RandPerm vector, using MemoryEfficientPSort
			SpParHelper::Print("Renamed Vertices\n");
	
			PSpMat_Int64 * G = new PSpMat_Int64(*DEL, false); // conversion from distributed edge list, keep self-loops
			SpParHelper::Print("Created Int64 Sparse Matrix\n");
			G->Reduce(degrees, Row, plus<int64_t>(), 0);	// identity is 0 
			delete G;
			SpParHelper::Print("Calculated degrees\n");

			// Start Kernel #1
			MPI::COMM_WORLD.Barrier();
			double t1 = MPI_Wtime();

			// the following constructor distributes edges to their rightful owners as well
			A = PSpMat_Bool(*DEL);	// remove self loops and duplicates (since this is of type boolean)
			delete DEL;	// free memory before symmetricizing
			
			PSpMat_Bool AT = A;
			AT.Transpose();
			A += AT;
			
			MPI::COMM_WORLD.Barrier();
			double t2=MPI_Wtime();
			
			if(myrank == 0)
				fprintf(stdout, "%.6lf seconds elapsed for Kernel #1\n", t2-t1);
		}
		A.PrintInfo();
		float balance = A.LoadImbalance();
		ostringstream outs;
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI_Wtime();

		FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid(), 0);
		A.Reduce(*ColSums, Column, plus<int64_t>(), 0); 	// plus<int64_t> matches the type of the output vector

		MPI::COMM_WORLD.Barrier();
		double t2=MPI_Wtime();
		if(myrank == 0)
			fprintf(stdout, "%.6lf seconds elapsed for getting column sums\n", t2-t1);

		FullyDistVec<int64_t, int64_t> Cands = ColSums->FindInds(bind2nd(greater<int64_t>(), 1));	// only the indices of non-isolated vertices
		delete ColSums;
		SpParHelper::Print("Found non-isolated vertices\n");	
	
		Cands.PrintInfo("Candidates array");
		FullyDistVec<int64_t,int64_t> First64(A.getcommgrid(), -1);
		Cands.RandPerm();
		Cands.PrintInfo("Candidates array (permuted)");
		First64.iota(64, 0);			
		//First64.DebugPrint();
		Cands = Cands(First64);		
		//Cands.DebugPrint();
		Cands.PrintInfo("First 64 of candidates (randomly chosen) array");
	
		double MTEPS[64]; double INVMTEPS[64]; double TIMES[64]; double EDGES[64];
		for(int i=0; i<64; ++i)
		{
			// FullyDistVec (shared_ptr<CommGrid> grid, IT globallen, NT initval, NT id);
			FullyDistVec<int64_t, int64_t> parents ( A.getcommgrid(), A.getncol(), (int64_t) -1, (int64_t) -1);	// identity is -1

			// FullyDistSpVec ( shared_ptr<CommGrid> grid, IT glen);
			FullyDistSpVec<int64_t, int64_t> fringe(A.getcommgrid(), A.getncol());	// numerical values are stored 0-based

			MPI::COMM_WORLD.Barrier();
			double t1 = MPI_Wtime();
			MPI_Pcontrol(1,"BFS");

			fringe.SetElement(Cands[i], Cands[i]);
			int iterations = 0;
			while(fringe.getnnz() > 0)
			{
				fringe.setNumToInd();
				//fringe.PrintInfo("fringe before SpMV");
				fringe = SpMV<SR>(A, fringe);	// SpMV with sparse vector
				// fringe.PrintInfo("fringe after SpMV");
				fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
				// fringe.PrintInfo("fringe after cleanup");
				parents += fringe;
				// parents.PrintInfo("Parents after addition");
				// SpParHelper::Print("Iteration finished\n");
				iterations++;
				MPI::COMM_WORLD.Barrier();
			}

			MPI_Pcontrol(-1,"BFS");
			MPI::COMM_WORLD.Barrier();
			double t2 = MPI_Wtime();

			FullyDistSpVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
			parentsp.Apply(set<int64_t>(1));

			// we use degrees on the directed graph, so that we don't count the reverse edges in the teps score
			int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
	
			ostringstream outnew;
			outnew << i << "th starting vertex was " << Cands[i] << endl;
			outnew << "Number iterations: " << iterations << endl;
			outnew << "Number of vertices found: " << parentsp.Reduce(plus<int64_t>(), (int64_t) 0) << endl; 
			outnew << "Number of edges traversed: " << nedges << endl;
			outnew << "BFS time: " << t2-t1 << " seconds" << endl;
			outnew << "MTEPS: " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
			TIMES[i] = t2-t1;
			EDGES[i] = nedges;
			MTEPS[i] = static_cast<double>(nedges) / (t2-t1) / 1000000.0;
			SpParHelper::Print(outnew.str());
		}
		SpParHelper::Print("Finished\n");
		ostringstream os;

		sort(EDGES, EDGES+64);
		os << "--------------------------" << endl;
		os << "Min nedges: " << EDGES[0] << endl;
		os << "First Quartile nedges: " << (EDGES[15] + EDGES[16])/2 << endl;
		os << "Median nedges: " << (EDGES[31] + EDGES[32])/2 << endl;
		os << "Third Quartile nedges: " << (EDGES[47] + EDGES[48])/2 << endl;
		os << "Max nedges: " << EDGES[63] << endl;
 		double mean = accumulate( EDGES, EDGES+64, 0.0 )/ 64;
		vector<double> zero_mean(64);	// find distances to the mean
		transform(EDGES, EDGES+64, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
		// self inner-product is sum of sum of squares
		double deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   		deviation = sqrt( deviation / 63 );
   		os << "Mean nedges: " << mean << endl;
		os << "STDDEV nedges: " << deviation << endl;
		os << "--------------------------" << endl;

		sort(TIMES,TIMES+64);
		os << "Min time: " << TIMES[0] << " seconds" << endl;
		os << "First Quartile time: " << (TIMES[15] + TIMES[16])/2 << " seconds" << endl;
		os << "Median time: " << (TIMES[31] + TIMES[32])/2 << " seconds" << endl;
		os << "Third Quartile time: " << (TIMES[47] + TIMES[48])/2 << " seconds" << endl;
		os << "Max time: " << TIMES[63] << " seconds" << endl;
 		mean = accumulate( TIMES, TIMES+64, 0.0 )/ 64;
		transform(TIMES, TIMES+64, zero_mean.begin(), bind2nd( minus<double>(), mean )); 	
		deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   		deviation = sqrt( deviation / 63 );
   		os << "Mean time: " << mean << " seconds" << endl;
		os << "STDDEV time: " << deviation << " seconds" << endl;
		os << "--------------------------" << endl;

		sort(MTEPS, MTEPS+64);
		os << "Min MTEPS: " << MTEPS[0] << endl;
		os << "First Quartile MTEPS: " << (MTEPS[15] + MTEPS[16])/2 << endl;
		os << "Median MTEPS: " << (MTEPS[31] + MTEPS[32])/2 << endl;
		os << "Third Quartile MTEPS: " << (MTEPS[47] + MTEPS[48])/2 << endl;
		os << "Max MTEPS: " << MTEPS[63] << endl;
		transform(MTEPS, MTEPS+64, INVMTEPS, safemultinv<double>()); 	// returns inf for zero teps
		double hteps = 64.0 / accumulate(INVMTEPS, INVMTEPS+64, 0.0);	
		os << "Harmonic mean of MTEPS: " << hteps << endl;
		transform(INVMTEPS, INVMTEPS+64, zero_mean.begin(), bind2nd(minus<double>(), 1/hteps));
		deviation = inner_product( zero_mean.begin(),zero_mean.end(), zero_mean.begin(), 0.0 );
   		deviation = sqrt( deviation / 63 ) * (hteps*hteps);	// harmonic_std_dev
		os << "Harmonic standard deviation of MTEPS: " << deviation << endl;
		SpParHelper::Print(os.str());
	}
	MPI::Finalize();
	return 0;
}

