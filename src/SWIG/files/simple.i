%module simple
%{
    #define SWIG_FILE_WITH_INIT
    #include "simple.h"

%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *x, int sizex)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *y, int sizey)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *z, int sizez)}

#include "simple.h"
