# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:22:55 2019

@author: 林志伟
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import sys


def get_outlier_ten(R, outlier_fra):
    '''
    输入R
    输出 指示矩阵 0正常点 1 异常点
    按全矩阵判定异常
    '''
    m,n,t = R.shape
    I_outlier = np.zeros([m,n,t])
    rng = np.random.RandomState(42)
    x = []
    x_ind = []
    for i in range(m):
        for j in range(n):
            for k in range(t):
                if R[i][j][k] > 0:
                    x.append(R[i][j][k])
                    x_ind.append(i*n*t+j*t+k)

    x = np.array(x)
    x = x.reshape(-1,1)

    clf = IsolationForest(max_samples= len(x), random_state=rng, contamination=outlier_fra)
    clf.fit(x)
    y_pred_train = clf.predict(x)        
    for i in range(len(y_pred_train)):
        if y_pred_train[i] == -1:
            row = int(x_ind[i] / (n*t))
            col = int((x_ind[i] - row * n * t) / t)
            dep = int(x_ind[i] % t)
            I_outlier[row][col][dep] = 1
    return I_outlier


def get_outlier_tensor(tensor,outlier_fra):
    m,n,t = tensor.shape
    I_outlier_tensor = get_outlier_ten(tensor, outlier_fra)  
    filename = "Full_Time_I_outlier_fra_"+str(outlier_fra)+"-new.npy"
    np.save(filename, I_outlier_tensor)
    print ("============outliers_fraction %s DONE ==============" % outlier_fra)
    
    
def load_data(filedir):
    R = -np.ones([142,4500,64])
    with open(filedir)as fin:
        for line in fin:
            user_id, service_id, time, val = list(map(float, line.split()))
            user_id = int(user_id)
            service_id = int(service_id)
            time = int(time)
            R[user_id][service_id][time] = val
    print (R.shape)
    return R 


def main(outlier_fra, dataset):
    if dataset == "rt":
        filedir = "dataset2/rtdata.txt"
    elif dataset == "tp":
        filedir = "dataset2/tpdata.txt"
    R = load_data(filedir)
    get_outlier_tensor(R, outlier_fra)

    
if __name__ == '__main__':
    outlier_fra = float(sys.argv[1])
    dataset = sys.argv[2]
    main(outlier_fra, dataset)    
    