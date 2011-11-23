# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.31
#
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _pyCombBLAS
import new
new_instancemethod = new.instancemethod
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'PySwigObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


class pySpParMat(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParMat, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParMat, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParMat(*args)
        try: self.this.append(this)
        except: self.this = this
    def getnnz(*args): return _pyCombBLAS.pySpParMat_getnnz(*args)
    def getnee(*args): return _pyCombBLAS.pySpParMat_getnee(*args)
    def getnrow(*args): return _pyCombBLAS.pySpParMat_getnrow(*args)
    def getncol(*args): return _pyCombBLAS.pySpParMat_getncol(*args)
    def load(*args): return _pyCombBLAS.pySpParMat_load(*args)
    def save(*args): return _pyCombBLAS.pySpParMat_save(*args)
    def GenGraph500Edges(*args): return _pyCombBLAS.pySpParMat_GenGraph500Edges(*args)
    def copy(*args): return _pyCombBLAS.pySpParMat_copy(*args)
    def __iadd__(*args): return _pyCombBLAS.pySpParMat___iadd__(*args)
    def assign(*args): return _pyCombBLAS.pySpParMat_assign(*args)
    def __mul__(*args): return _pyCombBLAS.pySpParMat___mul__(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParMat_SubsRef(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParMat___getitem__(*args)
    def removeSelfLoops(*args): return _pyCombBLAS.pySpParMat_removeSelfLoops(*args)
    def Apply(*args): return _pyCombBLAS.pySpParMat_Apply(*args)
    def DimWiseApply(*args): return _pyCombBLAS.pySpParMat_DimWiseApply(*args)
    def Prune(*args): return _pyCombBLAS.pySpParMat_Prune(*args)
    def Count(*args): return _pyCombBLAS.pySpParMat_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParMat_Reduce(*args)
    def Transpose(*args): return _pyCombBLAS.pySpParMat_Transpose(*args)
    def Find(*args): return _pyCombBLAS.pySpParMat_Find(*args)
    def SpMV(*args): return _pyCombBLAS.pySpParMat_SpMV(*args)
    def SpMV_inplace(*args): return _pyCombBLAS.pySpParMat_SpMV_inplace(*args)
    def Square(*args): return _pyCombBLAS.pySpParMat_Square(*args)
    def SpGEMM(*args): return _pyCombBLAS.pySpParMat_SpGEMM(*args)
    __swig_getmethods__["Column"] = lambda x: _pyCombBLAS.pySpParMat_Column
    if _newclass:Column = staticmethod(_pyCombBLAS.pySpParMat_Column)
    __swig_getmethods__["Row"] = lambda x: _pyCombBLAS.pySpParMat_Row
    if _newclass:Row = staticmethod(_pyCombBLAS.pySpParMat_Row)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParMat
    __del__ = lambda self : None;
pySpParMat_swigregister = _pyCombBLAS.pySpParMat_swigregister
pySpParMat_swigregister(pySpParMat)
pySpParMat_Column = _pyCombBLAS.pySpParMat_Column
pySpParMat_Row = _pyCombBLAS.pySpParMat_Row

class pySpParMatBool(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParMatBool, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParMatBool, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParMatBool(*args)
        try: self.this.append(this)
        except: self.this = this
    def getnnz(*args): return _pyCombBLAS.pySpParMatBool_getnnz(*args)
    def getnee(*args): return _pyCombBLAS.pySpParMatBool_getnee(*args)
    def getnrow(*args): return _pyCombBLAS.pySpParMatBool_getnrow(*args)
    def getncol(*args): return _pyCombBLAS.pySpParMatBool_getncol(*args)
    def load(*args): return _pyCombBLAS.pySpParMatBool_load(*args)
    def save(*args): return _pyCombBLAS.pySpParMatBool_save(*args)
    def GenGraph500Edges(*args): return _pyCombBLAS.pySpParMatBool_GenGraph500Edges(*args)
    def copy(*args): return _pyCombBLAS.pySpParMatBool_copy(*args)
    def __iadd__(*args): return _pyCombBLAS.pySpParMatBool___iadd__(*args)
    def assign(*args): return _pyCombBLAS.pySpParMatBool_assign(*args)
    def SpGEMM(*args): return _pyCombBLAS.pySpParMatBool_SpGEMM(*args)
    def __mul__(*args): return _pyCombBLAS.pySpParMatBool___mul__(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParMatBool_SubsRef(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParMatBool___getitem__(*args)
    def removeSelfLoops(*args): return _pyCombBLAS.pySpParMatBool_removeSelfLoops(*args)
    def Apply(*args): return _pyCombBLAS.pySpParMatBool_Apply(*args)
    def Prune(*args): return _pyCombBLAS.pySpParMatBool_Prune(*args)
    def Count(*args): return _pyCombBLAS.pySpParMatBool_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParMatBool_Reduce(*args)
    def Transpose(*args): return _pyCombBLAS.pySpParMatBool_Transpose(*args)
    def Find(*args): return _pyCombBLAS.pySpParMatBool_Find(*args)
    def SpMV(*args): return _pyCombBLAS.pySpParMatBool_SpMV(*args)
    def SpMV_inplace(*args): return _pyCombBLAS.pySpParMatBool_SpMV_inplace(*args)
    def Square(*args): return _pyCombBLAS.pySpParMatBool_Square(*args)
    __swig_getmethods__["Column"] = lambda x: _pyCombBLAS.pySpParMatBool_Column
    if _newclass:Column = staticmethod(_pyCombBLAS.pySpParMatBool_Column)
    __swig_getmethods__["Row"] = lambda x: _pyCombBLAS.pySpParMatBool_Row
    if _newclass:Row = staticmethod(_pyCombBLAS.pySpParMatBool_Row)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParMatBool
    __del__ = lambda self : None;
pySpParMatBool_swigregister = _pyCombBLAS.pySpParMatBool_swigregister
pySpParMatBool_swigregister(pySpParMatBool)
pySpParMatBool_Column = _pyCombBLAS.pySpParMatBool_Column
pySpParMatBool_Row = _pyCombBLAS.pySpParMatBool_Row

class pySpParMatObj1(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParMatObj1, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParMatObj1, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParMatObj1(*args)
        try: self.this.append(this)
        except: self.this = this
    def getnee(*args): return _pyCombBLAS.pySpParMatObj1_getnee(*args)
    def getnrow(*args): return _pyCombBLAS.pySpParMatObj1_getnrow(*args)
    def getncol(*args): return _pyCombBLAS.pySpParMatObj1_getncol(*args)
    def load(*args): return _pyCombBLAS.pySpParMatObj1_load(*args)
    def save(*args): return _pyCombBLAS.pySpParMatObj1_save(*args)
    def copy(*args): return _pyCombBLAS.pySpParMatObj1_copy(*args)
    def assign(*args): return _pyCombBLAS.pySpParMatObj1_assign(*args)
    def SpGEMM(*args): return _pyCombBLAS.pySpParMatObj1_SpGEMM(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParMatObj1_SubsRef(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParMatObj1___getitem__(*args)
    def removeSelfLoops(*args): return _pyCombBLAS.pySpParMatObj1_removeSelfLoops(*args)
    def Apply(*args): return _pyCombBLAS.pySpParMatObj1_Apply(*args)
    def DimWiseApply(*args): return _pyCombBLAS.pySpParMatObj1_DimWiseApply(*args)
    def Prune(*args): return _pyCombBLAS.pySpParMatObj1_Prune(*args)
    def Count(*args): return _pyCombBLAS.pySpParMatObj1_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParMatObj1_Reduce(*args)
    def Transpose(*args): return _pyCombBLAS.pySpParMatObj1_Transpose(*args)
    def Find(*args): return _pyCombBLAS.pySpParMatObj1_Find(*args)
    def SpMV(*args): return _pyCombBLAS.pySpParMatObj1_SpMV(*args)
    def Square(*args): return _pyCombBLAS.pySpParMatObj1_Square(*args)
    __swig_getmethods__["Column"] = lambda x: _pyCombBLAS.pySpParMatObj1_Column
    if _newclass:Column = staticmethod(_pyCombBLAS.pySpParMatObj1_Column)
    __swig_getmethods__["Row"] = lambda x: _pyCombBLAS.pySpParMatObj1_Row
    if _newclass:Row = staticmethod(_pyCombBLAS.pySpParMatObj1_Row)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParMatObj1
    __del__ = lambda self : None;
pySpParMatObj1_swigregister = _pyCombBLAS.pySpParMatObj1_swigregister
pySpParMatObj1_swigregister(pySpParMatObj1)
pySpParMatObj1_Column = _pyCombBLAS.pySpParMatObj1_Column
pySpParMatObj1_Row = _pyCombBLAS.pySpParMatObj1_Row

class pySpParMatObj2(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParMatObj2, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParMatObj2, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParMatObj2(*args)
        try: self.this.append(this)
        except: self.this = this
    def getnee(*args): return _pyCombBLAS.pySpParMatObj2_getnee(*args)
    def getnrow(*args): return _pyCombBLAS.pySpParMatObj2_getnrow(*args)
    def getncol(*args): return _pyCombBLAS.pySpParMatObj2_getncol(*args)
    def load(*args): return _pyCombBLAS.pySpParMatObj2_load(*args)
    def save(*args): return _pyCombBLAS.pySpParMatObj2_save(*args)
    def copy(*args): return _pyCombBLAS.pySpParMatObj2_copy(*args)
    def assign(*args): return _pyCombBLAS.pySpParMatObj2_assign(*args)
    def SpGEMM(*args): return _pyCombBLAS.pySpParMatObj2_SpGEMM(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParMatObj2_SubsRef(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParMatObj2___getitem__(*args)
    def removeSelfLoops(*args): return _pyCombBLAS.pySpParMatObj2_removeSelfLoops(*args)
    def Apply(*args): return _pyCombBLAS.pySpParMatObj2_Apply(*args)
    def DimWiseApply(*args): return _pyCombBLAS.pySpParMatObj2_DimWiseApply(*args)
    def Prune(*args): return _pyCombBLAS.pySpParMatObj2_Prune(*args)
    def Count(*args): return _pyCombBLAS.pySpParMatObj2_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParMatObj2_Reduce(*args)
    def Transpose(*args): return _pyCombBLAS.pySpParMatObj2_Transpose(*args)
    def Find(*args): return _pyCombBLAS.pySpParMatObj2_Find(*args)
    def SpMV(*args): return _pyCombBLAS.pySpParMatObj2_SpMV(*args)
    def Square(*args): return _pyCombBLAS.pySpParMatObj2_Square(*args)
    __swig_getmethods__["Column"] = lambda x: _pyCombBLAS.pySpParMatObj2_Column
    if _newclass:Column = staticmethod(_pyCombBLAS.pySpParMatObj2_Column)
    __swig_getmethods__["Row"] = lambda x: _pyCombBLAS.pySpParMatObj2_Row
    if _newclass:Row = staticmethod(_pyCombBLAS.pySpParMatObj2_Row)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParMatObj2
    __del__ = lambda self : None;
pySpParMatObj2_swigregister = _pyCombBLAS.pySpParMatObj2_swigregister
pySpParMatObj2_swigregister(pySpParMatObj2)
pySpParMatObj2_Column = _pyCombBLAS.pySpParMatObj2_Column
pySpParMatObj2_Row = _pyCombBLAS.pySpParMatObj2_Row

class pySpParVec(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParVec, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParVec, name)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParVec(*args)
        try: self.this.append(this)
        except: self.this = this
    def dense(*args): return _pyCombBLAS.pySpParVec_dense(*args)
    def getnee(*args): return _pyCombBLAS.pySpParVec_getnee(*args)
    def getnnz(*args): return _pyCombBLAS.pySpParVec_getnnz(*args)
    def __len__(*args): return _pyCombBLAS.pySpParVec___len__(*args)
    def len(*args): return _pyCombBLAS.pySpParVec_len(*args)
    def __add__(*args): return _pyCombBLAS.pySpParVec___add__(*args)
    def __sub__(*args): return _pyCombBLAS.pySpParVec___sub__(*args)
    def __iadd__(*args): return _pyCombBLAS.pySpParVec___iadd__(*args)
    def __isub__(*args): return _pyCombBLAS.pySpParVec___isub__(*args)
    def copy(*args): return _pyCombBLAS.pySpParVec_copy(*args)
    def any(*args): return _pyCombBLAS.pySpParVec_any(*args)
    def all(*args): return _pyCombBLAS.pySpParVec_all(*args)
    def intersectSize(*args): return _pyCombBLAS.pySpParVec_intersectSize(*args)
    def printall(*args): return _pyCombBLAS.pySpParVec_printall(*args)
    def load(*args): return _pyCombBLAS.pySpParVec_load(*args)
    def save(*args): return _pyCombBLAS.pySpParVec_save(*args)
    def Count(*args): return _pyCombBLAS.pySpParVec_Count(*args)
    def Apply(*args): return _pyCombBLAS.pySpParVec_Apply(*args)
    def ApplyInd(*args): return _pyCombBLAS.pySpParVec_ApplyInd(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParVec_SubsRef(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParVec_Reduce(*args)
    def Sort(*args): return _pyCombBLAS.pySpParVec_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pySpParVec_TopK(*args)
    def setNumToInd(*args): return _pyCombBLAS.pySpParVec_setNumToInd(*args)
    __swig_getmethods__["zeros"] = lambda x: _pyCombBLAS.pySpParVec_zeros
    if _newclass:zeros = staticmethod(_pyCombBLAS.pySpParVec_zeros)
    __swig_getmethods__["range"] = lambda x: _pyCombBLAS.pySpParVec_range
    if _newclass:range = staticmethod(_pyCombBLAS.pySpParVec_range)
    def abs(*args): return _pyCombBLAS.pySpParVec_abs(*args)
    def __delitem__(*args): return _pyCombBLAS.pySpParVec___delitem__(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParVec___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pySpParVec___setitem__(*args)
    def __repr__(*args): return _pyCombBLAS.pySpParVec___repr__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParVec
    __del__ = lambda self : None;
pySpParVec_swigregister = _pyCombBLAS.pySpParVec_swigregister
pySpParVec_swigregister(pySpParVec)
pySpParVec_zeros = _pyCombBLAS.pySpParVec_zeros
pySpParVec_range = _pyCombBLAS.pySpParVec_range

EWiseMult_inplacefirst = _pyCombBLAS.EWiseMult_inplacefirst
class pySpParVecObj1(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParVecObj1, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParVecObj1, name)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParVecObj1(*args)
        try: self.this.append(this)
        except: self.this = this
    def dense(*args): return _pyCombBLAS.pySpParVecObj1_dense(*args)
    def getnee(*args): return _pyCombBLAS.pySpParVecObj1_getnee(*args)
    def __len__(*args): return _pyCombBLAS.pySpParVecObj1___len__(*args)
    def len(*args): return _pyCombBLAS.pySpParVecObj1_len(*args)
    def copy(*args): return _pyCombBLAS.pySpParVecObj1_copy(*args)
    def any(*args): return _pyCombBLAS.pySpParVecObj1_any(*args)
    def all(*args): return _pyCombBLAS.pySpParVecObj1_all(*args)
    def intersectSize(*args): return _pyCombBLAS.pySpParVecObj1_intersectSize(*args)
    def printall(*args): return _pyCombBLAS.pySpParVecObj1_printall(*args)
    def load(*args): return _pyCombBLAS.pySpParVecObj1_load(*args)
    def save(*args): return _pyCombBLAS.pySpParVecObj1_save(*args)
    def Count(*args): return _pyCombBLAS.pySpParVecObj1_Count(*args)
    def Apply(*args): return _pyCombBLAS.pySpParVecObj1_Apply(*args)
    def ApplyInd(*args): return _pyCombBLAS.pySpParVecObj1_ApplyInd(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParVecObj1_SubsRef(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParVecObj1_Reduce(*args)
    def Sort(*args): return _pyCombBLAS.pySpParVecObj1_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pySpParVecObj1_TopK(*args)
    def __delitem__(*args): return _pyCombBLAS.pySpParVecObj1___delitem__(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParVecObj1___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pySpParVecObj1___setitem__(*args)
    def __repr__(*args): return _pyCombBLAS.pySpParVecObj1___repr__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParVecObj1
    __del__ = lambda self : None;
pySpParVecObj1_swigregister = _pyCombBLAS.pySpParVecObj1_swigregister
pySpParVecObj1_swigregister(pySpParVecObj1)
EWiseMult = _pyCombBLAS.EWiseMult

class pySpParVecObj2(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pySpParVecObj2, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pySpParVecObj2, name)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pySpParVecObj2(*args)
        try: self.this.append(this)
        except: self.this = this
    def dense(*args): return _pyCombBLAS.pySpParVecObj2_dense(*args)
    def getnee(*args): return _pyCombBLAS.pySpParVecObj2_getnee(*args)
    def __len__(*args): return _pyCombBLAS.pySpParVecObj2___len__(*args)
    def len(*args): return _pyCombBLAS.pySpParVecObj2_len(*args)
    def copy(*args): return _pyCombBLAS.pySpParVecObj2_copy(*args)
    def any(*args): return _pyCombBLAS.pySpParVecObj2_any(*args)
    def all(*args): return _pyCombBLAS.pySpParVecObj2_all(*args)
    def intersectSize(*args): return _pyCombBLAS.pySpParVecObj2_intersectSize(*args)
    def printall(*args): return _pyCombBLAS.pySpParVecObj2_printall(*args)
    def load(*args): return _pyCombBLAS.pySpParVecObj2_load(*args)
    def save(*args): return _pyCombBLAS.pySpParVecObj2_save(*args)
    def Count(*args): return _pyCombBLAS.pySpParVecObj2_Count(*args)
    def Apply(*args): return _pyCombBLAS.pySpParVecObj2_Apply(*args)
    def ApplyInd(*args): return _pyCombBLAS.pySpParVecObj2_ApplyInd(*args)
    def SubsRef(*args): return _pyCombBLAS.pySpParVecObj2_SubsRef(*args)
    def Reduce(*args): return _pyCombBLAS.pySpParVecObj2_Reduce(*args)
    def Sort(*args): return _pyCombBLAS.pySpParVecObj2_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pySpParVecObj2_TopK(*args)
    def __delitem__(*args): return _pyCombBLAS.pySpParVecObj2___delitem__(*args)
    def __getitem__(*args): return _pyCombBLAS.pySpParVecObj2___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pySpParVecObj2___setitem__(*args)
    def __repr__(*args): return _pyCombBLAS.pySpParVecObj2___repr__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pySpParVecObj2
    __del__ = lambda self : None;
pySpParVecObj2_swigregister = _pyCombBLAS.pySpParVecObj2_swigregister
pySpParVecObj2_swigregister(pySpParVecObj2)

class pyDenseParVec(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pyDenseParVec, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pyDenseParVec, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pyDenseParVec(*args)
        try: self.this.append(this)
        except: self.this = this
    def sparse(*args): return _pyCombBLAS.pyDenseParVec_sparse(*args)
    def len(*args): return _pyCombBLAS.pyDenseParVec_len(*args)
    def __len__(*args): return _pyCombBLAS.pyDenseParVec___len__(*args)
    def add(*args): return _pyCombBLAS.pyDenseParVec_add(*args)
    def __imul__(*args): return _pyCombBLAS.pyDenseParVec___imul__(*args)
    def __mul__(*args): return _pyCombBLAS.pyDenseParVec___mul__(*args)
    def __eq__(*args): return _pyCombBLAS.pyDenseParVec___eq__(*args)
    def __ne__(*args): return _pyCombBLAS.pyDenseParVec___ne__(*args)
    def copy(*args): return _pyCombBLAS.pyDenseParVec_copy(*args)
    def SubsRef(*args): return _pyCombBLAS.pyDenseParVec_SubsRef(*args)
    def RandPerm(*args): return _pyCombBLAS.pyDenseParVec_RandPerm(*args)
    def Sort(*args): return _pyCombBLAS.pyDenseParVec_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pyDenseParVec_TopK(*args)
    def printall(*args): return _pyCombBLAS.pyDenseParVec_printall(*args)
    def getnee(*args): return _pyCombBLAS.pyDenseParVec_getnee(*args)
    def getnnz(*args): return _pyCombBLAS.pyDenseParVec_getnnz(*args)
    def getnz(*args): return _pyCombBLAS.pyDenseParVec_getnz(*args)
    def any(*args): return _pyCombBLAS.pyDenseParVec_any(*args)
    def load(*args): return _pyCombBLAS.pyDenseParVec_load(*args)
    def save(*args): return _pyCombBLAS.pyDenseParVec_save(*args)
    def Count(*args): return _pyCombBLAS.pyDenseParVec_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pyDenseParVec_Reduce(*args)
    def Find(*args): return _pyCombBLAS.pyDenseParVec_Find(*args)
    def FindInds(*args): return _pyCombBLAS.pyDenseParVec_FindInds(*args)
    def Apply(*args): return _pyCombBLAS.pyDenseParVec_Apply(*args)
    def ApplyMasked(*args): return _pyCombBLAS.pyDenseParVec_ApplyMasked(*args)
    def EWiseApply(*args): return _pyCombBLAS.pyDenseParVec_EWiseApply(*args)
    __swig_getmethods__["range"] = lambda x: _pyCombBLAS.pyDenseParVec_range
    if _newclass:range = staticmethod(_pyCombBLAS.pyDenseParVec_range)
    def abs(*args): return _pyCombBLAS.pyDenseParVec_abs(*args)
    def __iadd__(*args): return _pyCombBLAS.pyDenseParVec___iadd__(*args)
    def __add__(*args): return _pyCombBLAS.pyDenseParVec___add__(*args)
    def __isub__(*args): return _pyCombBLAS.pyDenseParVec___isub__(*args)
    def __sub__(*args): return _pyCombBLAS.pyDenseParVec___sub__(*args)
    def __and__(*args): return _pyCombBLAS.pyDenseParVec___and__(*args)
    def __getitem__(*args): return _pyCombBLAS.pyDenseParVec___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pyDenseParVec___setitem__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pyDenseParVec
    __del__ = lambda self : None;
pyDenseParVec_swigregister = _pyCombBLAS.pyDenseParVec_swigregister
pyDenseParVec_swigregister(pyDenseParVec)
EWiseApply = _pyCombBLAS.EWiseApply
pyDenseParVec_range = _pyCombBLAS.pyDenseParVec_range

class pyDenseParVecObj1(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pyDenseParVecObj1, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pyDenseParVecObj1, name)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pyDenseParVecObj1(*args)
        try: self.this.append(this)
        except: self.this = this
    def sparse(*args): return _pyCombBLAS.pyDenseParVecObj1_sparse(*args)
    def len(*args): return _pyCombBLAS.pyDenseParVecObj1_len(*args)
    def __len__(*args): return _pyCombBLAS.pyDenseParVecObj1___len__(*args)
    def copy(*args): return _pyCombBLAS.pyDenseParVecObj1_copy(*args)
    def SubsRef(*args): return _pyCombBLAS.pyDenseParVecObj1_SubsRef(*args)
    def RandPerm(*args): return _pyCombBLAS.pyDenseParVecObj1_RandPerm(*args)
    def Sort(*args): return _pyCombBLAS.pyDenseParVecObj1_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pyDenseParVecObj1_TopK(*args)
    def printall(*args): return _pyCombBLAS.pyDenseParVecObj1_printall(*args)
    def getnee(*args): return _pyCombBLAS.pyDenseParVecObj1_getnee(*args)
    def load(*args): return _pyCombBLAS.pyDenseParVecObj1_load(*args)
    def save(*args): return _pyCombBLAS.pyDenseParVecObj1_save(*args)
    def Count(*args): return _pyCombBLAS.pyDenseParVecObj1_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pyDenseParVecObj1_Reduce(*args)
    def Find(*args): return _pyCombBLAS.pyDenseParVecObj1_Find(*args)
    def FindInds(*args): return _pyCombBLAS.pyDenseParVecObj1_FindInds(*args)
    def Apply(*args): return _pyCombBLAS.pyDenseParVecObj1_Apply(*args)
    def ApplyMasked(*args): return _pyCombBLAS.pyDenseParVecObj1_ApplyMasked(*args)
    def EWiseApply(*args): return _pyCombBLAS.pyDenseParVecObj1_EWiseApply(*args)
    def __getitem__(*args): return _pyCombBLAS.pyDenseParVecObj1___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pyDenseParVecObj1___setitem__(*args)
    def __repr__(*args): return _pyCombBLAS.pyDenseParVecObj1___repr__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pyDenseParVecObj1
    __del__ = lambda self : None;
pyDenseParVecObj1_swigregister = _pyCombBLAS.pyDenseParVecObj1_swigregister
pyDenseParVecObj1_swigregister(pyDenseParVecObj1)

class pyDenseParVecObj2(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pyDenseParVecObj2, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pyDenseParVecObj2, name)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_pyDenseParVecObj2(*args)
        try: self.this.append(this)
        except: self.this = this
    def sparse(*args): return _pyCombBLAS.pyDenseParVecObj2_sparse(*args)
    def len(*args): return _pyCombBLAS.pyDenseParVecObj2_len(*args)
    def __len__(*args): return _pyCombBLAS.pyDenseParVecObj2___len__(*args)
    def copy(*args): return _pyCombBLAS.pyDenseParVecObj2_copy(*args)
    def SubsRef(*args): return _pyCombBLAS.pyDenseParVecObj2_SubsRef(*args)
    def RandPerm(*args): return _pyCombBLAS.pyDenseParVecObj2_RandPerm(*args)
    def Sort(*args): return _pyCombBLAS.pyDenseParVecObj2_Sort(*args)
    def TopK(*args): return _pyCombBLAS.pyDenseParVecObj2_TopK(*args)
    def printall(*args): return _pyCombBLAS.pyDenseParVecObj2_printall(*args)
    def getnee(*args): return _pyCombBLAS.pyDenseParVecObj2_getnee(*args)
    def load(*args): return _pyCombBLAS.pyDenseParVecObj2_load(*args)
    def save(*args): return _pyCombBLAS.pyDenseParVecObj2_save(*args)
    def Count(*args): return _pyCombBLAS.pyDenseParVecObj2_Count(*args)
    def Reduce(*args): return _pyCombBLAS.pyDenseParVecObj2_Reduce(*args)
    def Find(*args): return _pyCombBLAS.pyDenseParVecObj2_Find(*args)
    def FindInds(*args): return _pyCombBLAS.pyDenseParVecObj2_FindInds(*args)
    def Apply(*args): return _pyCombBLAS.pyDenseParVecObj2_Apply(*args)
    def ApplyMasked(*args): return _pyCombBLAS.pyDenseParVecObj2_ApplyMasked(*args)
    def EWiseApply(*args): return _pyCombBLAS.pyDenseParVecObj2_EWiseApply(*args)
    def __getitem__(*args): return _pyCombBLAS.pyDenseParVecObj2___getitem__(*args)
    def __setitem__(*args): return _pyCombBLAS.pyDenseParVecObj2___setitem__(*args)
    def __repr__(*args): return _pyCombBLAS.pyDenseParVecObj2___repr__(*args)
    __swig_destroy__ = _pyCombBLAS.delete_pyDenseParVecObj2
    __del__ = lambda self : None;
pyDenseParVecObj2_swigregister = _pyCombBLAS.pyDenseParVecObj2_swigregister
pyDenseParVecObj2_swigregister(pyDenseParVecObj2)

class UnaryFunction(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnaryFunction, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UnaryFunction, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_UnaryFunction
    __del__ = lambda self : None;
    def __call__(*args): return _pyCombBLAS.UnaryFunction___call__(*args)
UnaryFunction_swigregister = _pyCombBLAS.UnaryFunction_swigregister
UnaryFunction_swigregister(UnaryFunction)

set = _pyCombBLAS.set
identity = _pyCombBLAS.identity
safemultinv = _pyCombBLAS.safemultinv
abs = _pyCombBLAS.abs
negate = _pyCombBLAS.negate
bitwise_not = _pyCombBLAS.bitwise_not
logical_not = _pyCombBLAS.logical_not
totality = _pyCombBLAS.totality
ifthenelse = _pyCombBLAS.ifthenelse
unary = _pyCombBLAS.unary
class BinaryFunction(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BinaryFunction, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BinaryFunction, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_BinaryFunction
    __del__ = lambda self : None;
    __swig_setmethods__["commutable"] = _pyCombBLAS.BinaryFunction_commutable_set
    __swig_getmethods__["commutable"] = _pyCombBLAS.BinaryFunction_commutable_get
    if _newclass:commutable = _swig_property(_pyCombBLAS.BinaryFunction_commutable_get, _pyCombBLAS.BinaryFunction_commutable_set)
    __swig_setmethods__["associative"] = _pyCombBLAS.BinaryFunction_associative_set
    __swig_getmethods__["associative"] = _pyCombBLAS.BinaryFunction_associative_get
    if _newclass:associative = _swig_property(_pyCombBLAS.BinaryFunction_associative_get, _pyCombBLAS.BinaryFunction_associative_set)
    def __call__(*args): return _pyCombBLAS.BinaryFunction___call__(*args)
BinaryFunction_swigregister = _pyCombBLAS.BinaryFunction_swigregister
BinaryFunction_swigregister(BinaryFunction)

plus = _pyCombBLAS.plus
minus = _pyCombBLAS.minus
multiplies = _pyCombBLAS.multiplies
divides = _pyCombBLAS.divides
modulus = _pyCombBLAS.modulus
fmod = _pyCombBLAS.fmod
pow = _pyCombBLAS.pow
max = _pyCombBLAS.max
min = _pyCombBLAS.min
bitwise_and = _pyCombBLAS.bitwise_and
bitwise_or = _pyCombBLAS.bitwise_or
bitwise_xor = _pyCombBLAS.bitwise_xor
logical_and = _pyCombBLAS.logical_and
logical_or = _pyCombBLAS.logical_or
logical_xor = _pyCombBLAS.logical_xor
equal_to = _pyCombBLAS.equal_to
not_equal_to = _pyCombBLAS.not_equal_to
greater = _pyCombBLAS.greater
less = _pyCombBLAS.less
greater_equal = _pyCombBLAS.greater_equal
less_equal = _pyCombBLAS.less_equal
binary = _pyCombBLAS.binary
binaryPtr = _pyCombBLAS.binaryPtr
bind1st = _pyCombBLAS.bind1st
bind2nd = _pyCombBLAS.bind2nd
compose1 = _pyCombBLAS.compose1
compose2 = _pyCombBLAS.compose2
not1 = _pyCombBLAS.not1
not2 = _pyCombBLAS.not2
class Semiring(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Semiring, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Semiring, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_Semiring(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _pyCombBLAS.delete_Semiring
    __del__ = lambda self : None;
    def mpi_op(*args): return _pyCombBLAS.Semiring_mpi_op(*args)
    def add(*args): return _pyCombBLAS.Semiring_add(*args)
    def multiply(*args): return _pyCombBLAS.Semiring_multiply(*args)
    def axpy(*args): return _pyCombBLAS.Semiring_axpy(*args)
Semiring_swigregister = _pyCombBLAS.Semiring_swigregister
Semiring_swigregister(Semiring)

TimesPlusSemiring = _pyCombBLAS.TimesPlusSemiring
SecondMaxSemiring = _pyCombBLAS.SecondMaxSemiring
class Obj1(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Obj1, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Obj1, name)
    __swig_setmethods__["weight"] = _pyCombBLAS.Obj1_weight_set
    __swig_getmethods__["weight"] = _pyCombBLAS.Obj1_weight_get
    if _newclass:weight = _swig_property(_pyCombBLAS.Obj1_weight_get, _pyCombBLAS.Obj1_weight_set)
    __swig_setmethods__["category"] = _pyCombBLAS.Obj1_category_set
    __swig_getmethods__["category"] = _pyCombBLAS.Obj1_category_get
    if _newclass:category = _swig_property(_pyCombBLAS.Obj1_category_get, _pyCombBLAS.Obj1_category_set)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_Obj1(*args)
        try: self.this.append(this)
        except: self.this = this
    def __repr__(*args): return _pyCombBLAS.Obj1___repr__(*args)
    def __eq__(*args): return _pyCombBLAS.Obj1___eq__(*args)
    def __ne__(*args): return _pyCombBLAS.Obj1___ne__(*args)
    def __lt__(*args): return _pyCombBLAS.Obj1___lt__(*args)
    __swig_setmethods__["hasPassedFilter"] = _pyCombBLAS.Obj1_hasPassedFilter_set
    __swig_getmethods__["hasPassedFilter"] = _pyCombBLAS.Obj1_hasPassedFilter_get
    if _newclass:hasPassedFilter = _swig_property(_pyCombBLAS.Obj1_hasPassedFilter_get, _pyCombBLAS.Obj1_hasPassedFilter_set)
    __swig_destroy__ = _pyCombBLAS.delete_Obj1
    __del__ = lambda self : None;
Obj1_swigregister = _pyCombBLAS.Obj1_swigregister
Obj1_swigregister(Obj1)

class Obj2(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Obj2, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Obj2, name)
    __swig_setmethods__["weight"] = _pyCombBLAS.Obj2_weight_set
    __swig_getmethods__["weight"] = _pyCombBLAS.Obj2_weight_get
    if _newclass:weight = _swig_property(_pyCombBLAS.Obj2_weight_get, _pyCombBLAS.Obj2_weight_set)
    __swig_setmethods__["category"] = _pyCombBLAS.Obj2_category_set
    __swig_getmethods__["category"] = _pyCombBLAS.Obj2_category_get
    if _newclass:category = _swig_property(_pyCombBLAS.Obj2_category_get, _pyCombBLAS.Obj2_category_set)
    def __init__(self, *args): 
        this = _pyCombBLAS.new_Obj2(*args)
        try: self.this.append(this)
        except: self.this = this
    def __repr__(*args): return _pyCombBLAS.Obj2___repr__(*args)
    def __eq__(*args): return _pyCombBLAS.Obj2___eq__(*args)
    def __ne__(*args): return _pyCombBLAS.Obj2___ne__(*args)
    def __lt__(*args): return _pyCombBLAS.Obj2___lt__(*args)
    __swig_setmethods__["hasPassedFilter"] = _pyCombBLAS.Obj2_hasPassedFilter_set
    __swig_getmethods__["hasPassedFilter"] = _pyCombBLAS.Obj2_hasPassedFilter_get
    if _newclass:hasPassedFilter = _swig_property(_pyCombBLAS.Obj2_hasPassedFilter_get, _pyCombBLAS.Obj2_hasPassedFilter_set)
    __swig_destroy__ = _pyCombBLAS.delete_Obj2
    __del__ = lambda self : None;
Obj2_swigregister = _pyCombBLAS.Obj2_swigregister
Obj2_swigregister(Obj2)

class UnaryPredicateObj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnaryPredicateObj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UnaryPredicateObj, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_UnaryPredicateObj
    __del__ = lambda self : None;
UnaryPredicateObj_swigregister = _pyCombBLAS.UnaryPredicateObj_swigregister
UnaryPredicateObj_swigregister(UnaryPredicateObj)

class UnaryFunctionObj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, UnaryFunctionObj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, UnaryFunctionObj, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_UnaryFunctionObj
    __del__ = lambda self : None;
UnaryFunctionObj_swigregister = _pyCombBLAS.UnaryFunctionObj_swigregister
UnaryFunctionObj_swigregister(UnaryFunctionObj)

unaryObj = _pyCombBLAS.unaryObj
unaryObjPred = _pyCombBLAS.unaryObjPred
class BinaryFunctionObj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BinaryFunctionObj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BinaryFunctionObj, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_BinaryFunctionObj
    __del__ = lambda self : None;
    __swig_setmethods__["commutable"] = _pyCombBLAS.BinaryFunctionObj_commutable_set
    __swig_getmethods__["commutable"] = _pyCombBLAS.BinaryFunctionObj_commutable_get
    if _newclass:commutable = _swig_property(_pyCombBLAS.BinaryFunctionObj_commutable_get, _pyCombBLAS.BinaryFunctionObj_commutable_set)
    __swig_setmethods__["associative"] = _pyCombBLAS.BinaryFunctionObj_associative_set
    __swig_getmethods__["associative"] = _pyCombBLAS.BinaryFunctionObj_associative_get
    if _newclass:associative = _swig_property(_pyCombBLAS.BinaryFunctionObj_associative_get, _pyCombBLAS.BinaryFunctionObj_associative_set)
    def __call__(*args): return _pyCombBLAS.BinaryFunctionObj___call__(*args)
    def rettype2nd_call(*args): return _pyCombBLAS.BinaryFunctionObj_rettype2nd_call(*args)
BinaryFunctionObj_swigregister = _pyCombBLAS.BinaryFunctionObj_swigregister
BinaryFunctionObj_swigregister(BinaryFunctionObj)

class BinaryPredicateObj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, BinaryPredicateObj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, BinaryPredicateObj, name)
    def __init__(self): raise AttributeError, "No constructor defined"
    __repr__ = _swig_repr
    __swig_destroy__ = _pyCombBLAS.delete_BinaryPredicateObj
    __del__ = lambda self : None;
BinaryPredicateObj_swigregister = _pyCombBLAS.BinaryPredicateObj_swigregister
BinaryPredicateObj_swigregister(BinaryPredicateObj)

binaryObjPred = _pyCombBLAS.binaryObjPred
class SemiringObj(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SemiringObj, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SemiringObj, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_SemiringObj(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _pyCombBLAS.delete_SemiringObj
    __del__ = lambda self : None;
    def mpi_op(*args): return _pyCombBLAS.SemiringObj_mpi_op(*args)
SemiringObj_swigregister = _pyCombBLAS.SemiringObj_swigregister
SemiringObj_swigregister(SemiringObj)
binaryObj = _pyCombBLAS.binaryObj

finalize = _pyCombBLAS.finalize
root = _pyCombBLAS.root
_nprocs = _pyCombBLAS._nprocs
prnt = _pyCombBLAS.prnt
testFunc = _pyCombBLAS.testFunc
class EWiseArg(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EWiseArg, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EWiseArg, name)
    __repr__ = _swig_repr
    def __init__(self, *args): 
        this = _pyCombBLAS.new_EWiseArg(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _pyCombBLAS.delete_EWiseArg
    __del__ = lambda self : None;
EWiseArg_swigregister = _pyCombBLAS.EWiseArg_swigregister
EWiseArg_swigregister(EWiseArg)

EWise_Index = _pyCombBLAS.EWise_Index
EWise = _pyCombBLAS.EWise
Graph500VectorOps = _pyCombBLAS.Graph500VectorOps
#import atexit
#atexit.register(finalize)

try:
	import ObjMethods

	ObjMethods.defUserCallbacks((Obj1,Obj2))
except ImportError:
	print "Failed to import ObjMethods!"
	print "----------------------------"
	print ""


EWise_OnlyNZ = _pyCombBLAS.EWise_OnlyNZ

