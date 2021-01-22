#include <algorithm>
#include <iostream>
using namespace std;

const double eps = 1e-10;

/* Perform the core approach of CTF */
void CTF(double *removedData, double *predData, int numUser, int numService, 
    int numTimeSlice, int dim, double gamma, double lmdau, double lmdas, double lmdat,
    int maxIter, double *Udata, double *Sdata, double *Tdata);

/* Update the corresponding X_hat */
void updateX_hat(bool flag, double ***X, double ***X_hat, double **U, double **S, 
    double **T, int numUser, int numService, int numTimeSlice, int dim);

/* Transform a vector into a matrix */ 
double **vector2Matrix(double *vector, int row, int col);

/* Transform a vector into a 3D tensor */ 
double ***vector2Tensor(double *vector, int row, int col, int height);

bool ***vector2Tensor(bool *vector, int row, int col, int height);

/* Get current date/time, format is YYYY-MM-DD hh:mm:ss */
const string currentDateTime();
