import time
import numpy as np
from utilities import *
cimport numpy as np # import C-API
from libcpp cimport bool


#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "CTF.h":
    void CTF(double *removedData, double *predData, int numUser, int numService, 
    int numTimeSlice, int dim, double gamma, double lmdau, double lmdas, double lmdat,
    int maxIter, double *Udata, double *Sdata, double *Tdata)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedTensor, para):
    cdef int numService = removedTensor.shape[1] 
    cdef int numUser = removedTensor.shape[0]
    cdef int numTimeSlice = removedTensor.shape[2]
    cdef int dim = para['dimension']
    cdef double gamma = para['gamma']
    cdef double lmdau = para['lambdau']
    cdef double lmdas = para['lambdas']
    cdef double lmdat = para['lambdat']
    cdef int maxIter = para['maxIter']

    # initialization
    cdef np.ndarray[double, ndim=2, mode='c'] U = np.random.rand(numUser, dim)        
    cdef np.ndarray[double, ndim=2, mode='c'] S = np.random.rand(numService, dim)
    cdef np.ndarray[double, ndim=2, mode='c'] T = np.random.rand(numTimeSlice, dim)
    cdef np.ndarray[double, ndim=3, mode='c'] predTensor =\
        np.zeros((numUser, numService, numTimeSlice))
    
    logger.info('Iterating...')
		
    # wrap up CTF.cpp
    CTF(<double *> (<np.ndarray[double, ndim=3, mode='c']> removedTensor).data,
        <double *> predTensor.data,
        numUser,
        numService,
        numTimeSlice,
        dim,
        gamma,
        lmdau,
        lmdas,
        lmdat,
        maxIter,
        <double *> U.data,
        <double *> S.data,
        <double *> T.data
        )

    return predTensor
#########################################################  
