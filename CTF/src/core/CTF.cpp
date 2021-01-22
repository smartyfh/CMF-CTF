#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include "CTF.h"
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;


/// note that predData is the output of this function
void CTF(double *removedData, double *predData, int numUser, int numService, 
    int numTimeSlice, int dim, double gamma, double lmdau, double lmdas, double lmdat,
    int maxIter, double *Udata, double *Sdata, double *Tdata)
{   
    // --- transfer the 1D pointer to 2D/3D array pointer
    double ***X = vector2Tensor(removedData, numUser, numService, numTimeSlice);
    double ***X_hat = vector2Tensor(predData, numUser, numService, numTimeSlice);
    double **U = vector2Matrix(Udata, numUser, dim);
    double **S = vector2Matrix(Sdata, numService, dim);
    double **T = vector2Matrix(Tdata, numTimeSlice, dim);
    bool ***I = vector2Tensor(new bool[numUser * numService * numTimeSlice], numUser, 
        numService, numTimeSlice);

    // --- compute indicator matrix I
    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            for (int k = 0; k < numTimeSlice; k++) {
                I[i][j][k] = fabs(X[i][j][k]) > eps;
            }
        }
    }
     
    // --- iterate by muplication rules
    double up, dw, res;
    double gamma2 = gamma * gamma;
    for (int iter = 0; iter < maxIter; iter++) {
        // update X_hat
        updateX_hat(false, X, X_hat, U, S, T, numUser, numService, numTimeSlice, dim);

        // update U
        int i, j, k, l;
        for (i = 0; i < numUser; i++) {          
            for (l = 0; l < dim; l++) {
                up = 0, dw = 0;
                for (j = 0; j < numService; j++) {
                    for (k = 0; k < numTimeSlice; k++) {
                        if (I[i][j][k]) {
                            res = X[i][j][k] - X_hat[i][j][k];
                            up += X[i][j][k] * (S[j][l] * T[k][l]) / (gamma2 + res * res);
                            dw += X_hat[i][j][k] * (S[j][l] * T[k][l]) / (gamma2 + res * res);
                        }
                    }
                }
                U[i][l] *= up / (dw + lmdau * U[i][l] + eps);
            } 
        }   

        // update X_hat
        updateX_hat(false, X, X_hat, U, S, T, numUser, numService, numTimeSlice, dim);

        // update S
        for (j = 0; j < numService; j++) {
            for (l = 0; l < dim; l++) {
                up = 0, dw = 0;
                for (i = 0; i < numUser; i++) {
                    for (k = 0; k < numTimeSlice; k++) {
                        if (I[i][j][k]) {
                            res = X[i][j][k] - X_hat[i][j][k];
                            up += X[i][j][k] * (U[i][l] * T[k][l]) / (gamma2 + res * res);
                            dw += X_hat[i][j][k] * (U[i][l] * T[k][l]) / (gamma2 + res * res);
                        }
                    }
                }
                S[j][l] *= up / (dw + lmdas * S[j][l] + eps);
            }
        }

        // update X_hat
        updateX_hat(false, X, X_hat, U, S, T, numUser, numService, numTimeSlice, dim);
            
        // update T
        for (k = 0; k < numTimeSlice; k++) {
            for (l = 0; l < dim; l++) {
                up = 0, dw = 0;
                for (i = 0; i < numUser; i++) {
                    for (j = 0; j < numService; j++) {
                        if (I[i][j][k]) {
                            res = X[i][j][k] - X_hat[i][j][k];
                            up += X[i][j][k] * (U[i][l] * S[j][l]) / (gamma2 + res * res);
                            dw += X_hat[i][j][k] * (U[i][l] * S[j][l]) / (gamma2 + res * res);
                        }
                    }
                }
                T[k][l] *= up / (dw + lmdat * T[k][l] + eps);
            }
        }  
    }

    // update X_hat
    updateX_hat(true, X, X_hat, U, S, T, numUser, numService, numTimeSlice, dim);
  
    delete ((char*) U);
    delete ((char*) S);
    delete ((char*) T);
    delete ((char*) X);
    delete ((char*) X_hat);
    delete ((char*) I);
}


void updateX_hat(bool flag, double ***X, double ***X_hat, double **U, double **S, 
    double **T, int numUser, int numService, int numTimeSlice, int dim)
{
    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            for (int k = 0; k < numTimeSlice; k++) {
                if (flag == true || X[i][j][k] > 0) {
                    X_hat[i][j][k] = 0;
                    for (int l = 0; l < dim; l++) {
                        X_hat[i][j][k] += U[i][l] * S[j][l] * T[k][l];
                    }
                }
            }
        }
    }
}


double **vector2Matrix(double *vector, int row, int col)  
{
    double **matrix = new double *[row];
    if (!matrix) {
        cout << "Memory allocation failed in vector2Matrix." << endl;
        return NULL;
    }

    int i;
    for (i = 0; i < row; i++) {
        matrix[i] = vector + i * col;  
    }
    return matrix;
}


double ***vector2Tensor(double *vector, int row, int col, int height)
{
    double ***tensor = new double **[row];
    if (!tensor) {
        cout << "Memory allocation failed in vector2Tensor." << endl;
        return NULL;
    }

    int i, j;
    for (i = 0; i < row; i++) {
        tensor[i] = new double *[col];
        if (!tensor[i]) {
            cout << "Memory allocation failed in vector2Tensor." << endl;
            return NULL;
        }

        for (j = 0; j < col; j++) {
            tensor[i][j] = vector + i * col * height + j * height;
        }
    }

    return tensor;
}


bool ***vector2Tensor(bool *vector, int row, int col, int height)
{
    bool ***tensor = new bool **[row];
    if (!tensor) {
        cout << "Memory allocation failed in vector2Tensor." << endl;
        return NULL;
    }
    
    int i, j;
    for (i = 0; i < row; i++) {
        tensor[i] = new bool *[col];
        if (!tensor[i]) {
            cout << "Memory allocation failed in vector2Tensor." << endl;
            return NULL;
        }
        
        for (j = 0; j < col; j++) {
            tensor[i][j] = vector + i * col * height + j * height;
        }
    }
    
    return tensor;
}


const string currentDateTime() 
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}
