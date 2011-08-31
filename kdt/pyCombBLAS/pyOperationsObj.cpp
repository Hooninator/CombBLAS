#include "pyOperationsObj.h"
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <Python.h>

namespace op{

/**************************\
| UNARY OPERATIONS
\**************************/


UnaryFunctionObj unaryObj(PyObject *pyfunc)
{
	return UnaryFunctionObj(pyfunc);
}


UnaryPredicateObj unaryObjPred(PyObject *pyfunc)
{
	return UnaryPredicateObj(pyfunc);
}


// Slightly un-standard ops:
#if 0
template<typename T>
struct set_s: public ConcreteUnaryFunction<T>
{
	set_s(T myvalue): value(myvalue) {};
	/** @returns value regardless of x */
	T operator()(const T& x) const
	{
		return value;
	} 
	T value;
};

UnaryFunction set(Obj2* val)
{
	return UnaryFunction(new set_s<Obj2>(Obj2(*val)));
}

UnaryFunction set(Obj1* val)
{
	return UnaryFunction(new set_s<Obj1>(Obj1(*val)));
}
#endif



/////////////////////////////////////////////////////


#if 0

/**************************\
| BINARY OPERATIONS
\**************************/

#define DECL_BINARY_STRUCT(name, operation) 							\
	template<typename T>												\
	struct name : public ConcreteBinaryFunction<T>						\
	{																	\
		T operator()(const T& x, const T& y) const						\
		{																\
			return operation;											\
		}																\
	};
	
#define DECL_BINARY_FUNC(structname, name, as, com, operation)			\
	DECL_BINARY_STRUCT(structname, operation)							\
	BinaryFunction name()												\
	{																	\
		return BinaryFunction(new structname<doubleint>(), as, com);	\
	}																


//// Custom Python callback
template<typename T>
struct binary_s: public ConcreteBinaryFunction<T>
{
	PyObject *pyfunc;

	binary_s(PyObject *pyfunc_in): pyfunc(pyfunc_in)
	{
		Py_INCREF(pyfunc);
	}
	
	~binary_s()
	{
		Py_DECREF(pyfunc);
	}
	
	T operator()(const T& x, const T& y) const
	{
		PyObject *arglist;
		PyObject *result;
		double dres = 0;
		
		arglist = Py_BuildValue("(d d)", static_cast<double>(x), static_cast<double>(y));    // Build argument list
		result = PyEval_CallObject(pyfunc,arglist);     // Call Python
		Py_DECREF(arglist);                             // Trash arglist
		if (result) {                                   // If no errors, return double
			dres = PyFloat_AsDouble(result);
		}
		Py_XDECREF(result);
		return T(dres);
	} 
};

BinaryFunction binary(PyObject *pyfunc)
{
	// assumed to be associative but not commutative
	return BinaryFunction(new binary_s<doubleint>(pyfunc), true, false);
}

/**************************\
| METHODS
\**************************/
BinaryFunction* BinaryFunction::currentlyApplied = NULL;
MPI_Op BinaryFunction::staticMPIop;
	
void BinaryFunction::apply(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
	doubleint* in = (doubleint*)invec;
	doubleint* inout = (doubleint*)inoutvec;
	
	for (int i = 0; i < *len; i++)
	{
		inout[i] = (*currentlyApplied)(in[i], inout[i]);
	}
}

MPI_Op* BinaryFunction::getMPIOp()
{
	//cout << "setting mpi op" << endl;
	if (currentlyApplied != NULL)
	{
		cout << "There is an internal error in creating a MPI version of a BinaryFunction: Conflict between two BFs." << endl;
		std::exit(1);
	}
	else if (currentlyApplied == this)
	{
		return &staticMPIop;
	}

	currentlyApplied = this;
	MPI_Op_create(BinaryFunction::apply, commutable, &staticMPIop);
	return &staticMPIop;
}

void BinaryFunction::releaseMPIOp()
{
	//cout << "free mpi op" << endl;

	if (currentlyApplied == this)
		currentlyApplied = NULL;
}
#endif

/**************************\
| SEMIRING
\**************************/
/*
template <>
Semiring* SemiringTemplArg<doubleint, doubleint>::currentlyApplied = NULL;

Semiring::Semiring(PyObject *add, PyObject *multiply)
	: type(CUSTOM), pyfunc_add(add), pyfunc_multiply(multiply), binfunc_add(&binary(add))
{
	Py_INCREF(pyfunc_add);
	Py_INCREF(pyfunc_multiply);
}
Semiring::~Semiring()
{
	Py_XDECREF(pyfunc_add);
	Py_XDECREF(pyfunc_multiply);
	assert((SemiringTemplArg<doubleint, doubleint>::currentlyApplied != this));
}

void Semiring::enableSemiring()
{
	if (SemiringTemplArg<doubleint, doubleint>::currentlyApplied != NULL)
	{
		cout << "There is an internal error in selecting a Semiring: Conflict between two Semirings." << endl;
		std::exit(1);
	}
	SemiringTemplArg<doubleint, doubleint>::currentlyApplied = this;
	binfunc_add->getMPIOp();
}

void Semiring::disableSemiring()
{
	binfunc_add->releaseMPIOp();
	SemiringTemplArg<doubleint, doubleint>::currentlyApplied = NULL;
}

doubleint Semiring::add(const doubleint & arg1, const doubleint & arg2)
{
	PyObject *arglist;
	PyObject *result;
	double dres = 0;
	
	arglist = Py_BuildValue("(d d)", arg1.d, arg2.d);    // Build argument list
	result = PyEval_CallObject(pyfunc_add, arglist);     // Call Python
	Py_DECREF(arglist);                                  // Trash arglist
	if (result) {                                        // If no errors, return double
		dres = PyFloat_AsDouble(result);
	}
	Py_XDECREF(result);
	return doubleint(dres);
}

doubleint Semiring::multiply(const doubleint & arg1, const doubleint & arg2)
{
	PyObject *arglist;
	PyObject *result;
	double dres = 0;
	
	arglist = Py_BuildValue("(d d)", arg1.d, arg2.d);         // Build argument list
	result = PyEval_CallObject(pyfunc_multiply, arglist);     // Call Python
	Py_DECREF(arglist);                                       // Trash arglist
	if (result) {                                             // If no errors, return double
		dres = PyFloat_AsDouble(result);
	}
	Py_XDECREF(result);
	return doubleint(dres);
}

void Semiring::axpy(doubleint a, const doubleint & x, doubleint & y)
{
	y = add(y, multiply(a, x));
}

Semiring TimesPlusSemiring()
{
	return Semiring(Semiring::TIMESPLUS);
}

Semiring MinPlusSemiring()
{
	return Semiring(Semiring::PLUSMIN);
}

Semiring SecondMaxSemiring()
{
	return Semiring(Semiring::SECONDMAX);
}
*/
} // namespace op
