
%module pyCombBLAS

//%typemap(in) int64_t = long long;
//%typemap(out) int64_t = long long;
%apply long long {int64_t}

// This block gets copied verbatim into the header area of the generated wrapper.

%{
#define SWIG_FILE_WITH_INIT

#include "pyCombBLAS.h"
%}


// This block gets called when the module is loaded. It is wrapped in extern "C".
%init %{
init_pyCombBLAS_MPI();
%}

// It is possible to have the generated python code also include some custom code.
// This may be a good place to add an atexit() to call mpi finalize.
%pragma(python) code="import atexit"
%pragma(python) code="atexit.register(DiGraph.finalize())"


// Language independent exception handler
%include exception.i    

%exception {
	try {
		$action
	} catch(string& stringReason) {
		const char* sData = (char*)stringReason.c_str();
		SWIG_exception(SWIG_RuntimeError,sData);
		SWIG_exception(SWIG_IndexError,sData);
	} catch(...) {
		SWIG_exception(SWIG_RuntimeError,"Unknown exception");
	}
}

// Grab a Python function object as a Python object.
// Based on example from SWIG docs: http://www.swig.org/Doc1.1/HTML/Python.html#n11
#ifdef SWIG<python>
%typemap(in) PyObject *pyfunc {
  if (!PyCallable_Check($input)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object!");
      return NULL;
  }
  $1 = $input;
}
#else
// Please define a way to handle callbacks in your target language.
#endif

// wrapped classes

class pySpParMat {
public:
	pySpParMat();
	pySpParMat(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);

public:
	int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	double GenGraph500Edges(int scale);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMat copy();
	pySpParMat& operator+=(const pySpParMat& other);
	pySpParMat& assign(const pySpParMat& other);
	pySpParMat SpMM(pySpParMat& other, op::Semiring* sring = NULL);
	pySpParMat operator*(pySpParMat& other);
	pySpParMat SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols);
	pySpParMat __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunction* f);
	void ColWiseApply(const pySpParVec& values, op::BinaryFunction* f);
	void Prune(op::UnaryFunction* f);
	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVec Reduce(int dim, op::BinaryFunction* f, double identity = 0);
	pyDenseParVec Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity = 0);
	
	void Transpose();
	//void EWiseMult(pySpParMat* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec SpMV_PlusTimes(const pySpParVec& x);
	pySpParVec SpMV_SelMax(const pySpParVec& x);
	void SpMV_SelMax_inplace(pySpParVec& x);

	pySpParVec SpMV(const pySpParVec& x, op::Semiring* sring);
	void SpMV_inplace(pySpParVec& x, op::Semiring* sring);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);
pySpParMat EWiseApply(const pySpParMat& A, const pySpParMat& B, op::BinaryFunction *bf, bool notB = false, double defaultBValue = 1);


class pySpParMatBool {
public:
	pySpParMatBool();
	pySpParMatBool(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVec* vals);
	
	pySpParMatBool(const pySpParMat& copyFrom);

public:
	int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	double GenGraph500Edges(int scale);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMatBool copy();
	pySpParMatBool& operator+=(const pySpParMatBool& other);
	pySpParMatBool& assign(const pySpParMatBool& other);
	pySpParMatBool SpMM(pySpParMatBool& other);
	pySpParMatBool operator*(pySpParMatBool& other);
	pySpParMatBool SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols);
	pySpParMatBool __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunction* f);
	void ColWiseApply(const pySpParVec& values, op::BinaryFunction* f);
	void Prune(op::UnaryFunction* f);
	int64_t Count(op::UnaryFunction* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVec Reduce(int dim, op::BinaryFunction* f, double identity = 0);
	pyDenseParVec Reduce(int dim, op::BinaryFunction* bf, op::UnaryFunction* uf, double identity = 0);
	
	void Transpose();
	//void EWiseMult(pySpParMatBool rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVec* outvals) const;
public:
	pySpParVec SpMV_PlusTimes(const pySpParVec& v);
	pySpParVec SpMV_SelMax(const pySpParVec& v);
	void SpMV_SelMax_inplace(pySpParVec& v);
	
public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

pySpParMatBool EWiseMult(const pySpParMatBool& A1, const pySpParMatBool& A2, bool exclude);
pySpParMatBool EWiseApply(const pySpParMatBool& A, const pySpParMatBool& B, op::BinaryFunction *bf, bool notB = false, double defaultBValue = 1);


class pySpParVec {
public:
	pySpParVec(int64_t length);
	
	pyDenseParVec dense() const;

public:
	int64_t getnee() const;
	int64_t getnnz() const;
	int64_t __len__() const;
	int64_t len() const;

	pySpParVec operator+(const pySpParVec& other);
	pySpParVec operator-(const pySpParVec& other);
	pySpParVec operator+(const pyDenseParVec& other);
	pySpParVec operator-(const pyDenseParVec& other);

	pySpParVec& operator+=(const pySpParVec& other);
	pySpParVec& operator-=(const pySpParVec& other);
	pySpParVec& operator+=(const pyDenseParVec& other);
	pySpParVec& operator-=(const pyDenseParVec& other);
	pySpParVec copy();

public:	
	bool any() const; // any nonzeros
	bool all() const; // all nonzeros
	
	int64_t intersectSize(const pySpParVec& other);
	
	void printall();
	
public:	
	void load(const char* filename);

public:
	// The functions commented out here presently do not exist in CombBLAS
	int64_t Count(op::UnaryFunction* op);
	//pySpParVec Find(op::UnaryFunction* op);
	//pyDenseParVec FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	//void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);

	pyDenseParVec SubsRef(const pyDenseParVec& ri);
	
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	
	pySpParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.
	
	void setNumToInd();

public:
	static pySpParVec zeros(int64_t howmany);
	static pySpParVec range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pySpParVec abs();
	void __delitem__(const pyDenseParVec& key);
	void __delitem__(int64_t key);
	
	double __getitem__(int64_t key);
	double __getitem__(double  key);
	pyDenseParVec __getitem__(const pyDenseParVec& key);
	
	void __setitem__(int64_t key, double value);
	void __setitem__(double  key, double value);
	void __setitem__(const pyDenseParVec& key, const pyDenseParVec& value);
	//void __setitem__(const pyDenseParVec& key, int64_t value);
	void __setitem__(const char* key, double value);	
	
	char* __repr__();

};

//      EWiseMult has 2 flavors:
//      - if Exclude is false, will do element-wise multiplication
//      - if Exclude is true, will remove from the result vector all elements
//          whose corresponding element of the second vector is "nonzero"
//          (i.e., not equal to the sparse vector's identity value)  '

//pySpParVec EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude);
pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);
void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero);



class pyDenseParVec {
public:
	pyDenseParVec(int64_t size, double init);
	pyDenseParVec(int64_t size, double init, double zero);
	
	pySpParVec sparse() const;
	pySpParVec sparse(double zero) const;
	
public:
	int64_t len() const;
	int64_t __len__() const;
	
	void add(const pyDenseParVec& other);
	void add(const pySpParVec& other);
	pyDenseParVec& operator+=(const pyDenseParVec & rhs);
	pyDenseParVec& operator-=(const pyDenseParVec & rhs);
	pyDenseParVec& operator+=(const pySpParVec & rhs);
	pyDenseParVec& operator-=(const pySpParVec & rhs);
	pyDenseParVec& operator*=(const pyDenseParVec& rhs);
	pyDenseParVec& operator*=(const pySpParVec& rhs);
	
	pyDenseParVec operator+(const pyDenseParVec & rhs);
	pyDenseParVec operator-(const pyDenseParVec & rhs);
	pyDenseParVec operator+(const pySpParVec & rhs);
	pyDenseParVec operator-(const pySpParVec & rhs);
	pyDenseParVec operator*(const pyDenseParVec& rhs);
	pyDenseParVec operator*(const pySpParVec& rhs);
	
	pyDenseParVec operator==(const pyDenseParVec& other);
	pyDenseParVec operator!=(const pyDenseParVec& other);

	pyDenseParVec copy();
	
	pyDenseParVec SubsRef(const pyDenseParVec& ri);

	void RandPerm(); // Randomly permutes the vector
	pyDenseParVec Sort(); // Does an in-place sort and returns the permutation used in the sort.
	pyDenseParVec TopK(int64_t k); // Returns a vector of the k largest elements.

	void printall();
	
public:
	
	int64_t getnee() const;
	int64_t getnnz() const;
	int64_t getnz() const;
	bool any() const;
	
public:	
	void load(const char* filename);
	
public:
	int64_t Count(op::UnaryFunction* op);
	double Reduce(op::BinaryFunction* f, op::UnaryFunction* uf = NULL);
	pySpParVec Find(op::UnaryFunction* op);
	pySpParVec __getitem__(op::UnaryFunction* op);
	pyDenseParVec FindInds(op::UnaryFunction* op);
	void Apply(op::UnaryFunction* op);
	void ApplyMasked(op::UnaryFunction* op, const pySpParVec& mask);
	void EWiseApply(const pyDenseParVec& other, op::BinaryFunction *f);
	void EWiseApply(const pySpParVec& other, op::BinaryFunction *f, bool doNulls = false, double nullValue = 0);

public:
	static pyDenseParVec range(int64_t howmany, int64_t start);
	
public:
	// Functions from PyCombBLAS
	pyDenseParVec abs();
	
	pyDenseParVec& operator+=(double value);
	pyDenseParVec operator+(double value);
	pyDenseParVec& operator-=(double value);
	pyDenseParVec operator-(double value);
	
	pyDenseParVec __and__(const pyDenseParVec& other);
	
	double __getitem__(int64_t key);
	double __getitem__(double  key);
	pyDenseParVec __getitem__(const pyDenseParVec& key);

	void __setitem__(int64_t key, double value);
	void __setitem__(double  key, double value);
	void __setitem__(const pySpParVec& key, const pySpParVec& value);
	void __setitem__(const pySpParVec& key, double value);
};

namespace op {

class UnaryFunction {

	protected:
	UnaryFunction(): op(NULL) {}
	public:
	~UnaryFunction() { /*delete op; op = NULL;*/ }
	
	doubleint operator()(const doubleint x) const
	{
		return (*op)(x);
	}
};

UnaryFunction set(double val);
UnaryFunction identity();
UnaryFunction safemultinv();
UnaryFunction abs();
UnaryFunction negate();
UnaryFunction bitwise_not();
UnaryFunction logical_not();
UnaryFunction totality();
UnaryFunction ifthenelse(UnaryFunction& predicate, UnaryFunction& runTrue, UnaryFunction& runFalse);

UnaryFunction unary(PyObject *pyfunc);

class BinaryFunction {
	protected:
	BinaryFunction(): op(NULL), commutable(false), associative(false) {}
	public:
	~BinaryFunction() { /*delete op; op = NULL;*/ }
	
	bool commutable;
	bool associative;
	
	doubleint operator()(const doubleint& x, const doubleint& y) const
	{
		return (*op)(x, y);
	}

};

BinaryFunction plus();
BinaryFunction minus();
BinaryFunction multiplies();
BinaryFunction divides();
BinaryFunction modulus();
BinaryFunction fmod();
BinaryFunction pow();

BinaryFunction max();
BinaryFunction min();

BinaryFunction bitwise_and();
BinaryFunction bitwise_or();
BinaryFunction bitwise_xor();
BinaryFunction logical_and();
BinaryFunction logical_or();
BinaryFunction logical_xor();

BinaryFunction equal_to();
BinaryFunction not_equal_to();
BinaryFunction greater();
BinaryFunction less();
BinaryFunction greater_equal();
BinaryFunction less_equal();

BinaryFunction binary(PyObject *pyfunc);

// Glue functions

UnaryFunction bind1st(BinaryFunction& op, double val);
UnaryFunction bind2nd(BinaryFunction& op, double val);
UnaryFunction compose1(UnaryFunction& f, UnaryFunction& g); // h(x) is the same as f(g(x))
UnaryFunction compose2(BinaryFunction& f, UnaryFunction& g1, UnaryFunction& g2); // h(x) is the same as f(g1(x), g2(x))
UnaryFunction not1(UnaryFunction& f);
BinaryFunction not2(BinaryFunction& f);



class Semiring {
	protected:
	Semiring(): pyfunc_add(NULL), pyfunc_multiply(NULL), binfunc_add(NULL) {}
	public:
	Semiring(PyObject *add, PyObject *multiply);
	~Semiring();
	
	MPI_Op mpi_op()
	{
		return *(binfunc_add->getMPIOp());
	}
	
	doubleint add(const doubleint & arg1, const doubleint & arg2);	
	doubleint multiply(const doubleint & arg1, const doubleint & arg2);
	void axpy(doubleint a, const doubleint & x, doubleint & y);

};

} // namespace op




void finalize();
bool root();
int _nprocs();

void testFunc(double (*f)(double));


