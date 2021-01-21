# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:22:26 2019

@author: 林志伟
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import sys


def get_outlier_mat(R,outliers_fraction):
    '''
    输入R
    输出 指示矩阵 0正常点 1 异常点
    按全矩阵判定异常
    '''
    m,n = R.shape
    I_outlier = np.zeros([m,n])
    rng = np.random.RandomState(42)
    x = []
    x_ind = []
    for i in range(m):
        for j in range(n):
            if R[i][j] >=0:
                x.append(R[i][j])
                x_ind.append(i*n+j)

    x = np.array(x)
    x = x.reshape(-1,1)

    clf = IsolationForest(max_samples= len(x), random_state=rng, contamination=outliers_fraction)
    clf.fit(x)
    y_pred_train = clf.predict(x)        
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            row = int(x_ind[i] / n)
            col = int(x_ind[i] % n)
            I_outlier[row][col] = 1
    
    filename = "Full_I_outlier_fra_"+str(outliers_fraction)+".npy"
    np.save(filename, I_outlier)
    print ("============outliers_fraction %s DONE ==============" % outliers_fraction)


    
def load_data(filedir):
    R = []
    with open(filedir)as fin:
        for line in fin:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R 


def main(outlier_fra):
    filedir = "dataset1/rtMatrix.txt"
    R = load_data(filedir)
    get_outlier_mat(R, outlier_fra)

    
if __name__ == '__main__':
    outlier_fra = float(sys.argv[1])
    main(outlier_fra)    
    