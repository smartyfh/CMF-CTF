# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:58:15 2019

@author: 林志伟
"""

#CMF 

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


def CMF(R,l,c,lamdau,lamdas,beta,iteration,decay_steps,decay_rate,I_train,I_test):
    #R        评分矩阵  m*n
    #l       矩阵分解维度    R= U*S^T    m*l, l*n
    #c       柯西估计器的参数
    #lamdau  矩阵U的正则项参数
    #lamdas  矩阵S的正则项参数
    #beta    梯度下降学习率    
    
    m,n = R.shape
    U = np.random.random((m,l))
    S = np.random.random((n,l))
    loss = []
    rmse = []
    mae = []
    for i in range(iteration):
        decayed_beta = beta * decay_rate**(i / decay_steps)
        P = R - np.dot(U,S.T)  
        cmat = np.ones((m,n))*c*c
        Pstart = P*P + cmat           
        H = I_train*P/Pstart
    
        Ud = lamdau*U -np.dot(H, S)
        Sd = lamdas*S - np.dot(H.T, U)
            
        U = U - decayed_beta*Ud
        S = S - decayed_beta*Sd

        los = np.sum(I_train*np.log(Pstart/cmat)) + \
                lamdau*np.linalg.norm(U) + lamdas*np.linalg.norm(S)
        loss.append(los)
        

        rms= np.sum(I_test *P*P) / np.sum(I_test)
        rms = math.sqrt(rms)
        rmse.append(rms)
        
        ma = np.sum(np.abs(I_test*P))/ np.sum(I_test)
        mae.append(ma)
    return U,S,loss,rmse,mae


def load_data(filedir):
    R = []
    with open(filedir)as fin:
        for line in fin:
            R.append(list(map(float, line.split())))
        R = np.array(R)
    return R        
            
            
def train_test(R,density):
    #density 划分训练集和测试集
    I = R.copy()
    I[I >= 0] = 1
    I[I < 0] = 0
    I_train = I.copy()
    for i in range(len(I)):
        for j in range(len(I[0])):
            if I_train[i][j] > 0:
                tmp = random.random()
                if tmp > density:
                    I_train[i][j] = 0
    I_test = I - I_train
    return I_train, I_test


def merge_test_outlier(I_outlier, I_test):
    I_merge = I_test.copy()
    for i in range(len(I_test)):
        for j in range(len(I_test[0])):
            if I_test[i][j]==1 and I_outlier[i][j]==1:
                I_merge[i][j] = 0
    #print ("原始的测试样本数%d" % np.sum(I_test))
    #print ("除去异常样本后测试样本数%d" % np.sum(I_merge))
    return I_merge

def main(density, outlier_fra, dataset):
    print ("Pram density=%s  outlier_fra=%s dataset= %s" % (density,outlier_fra,dataset))
    #outlier_fra = 0.1  #异常点的比例
    l = 30   #矩阵分解的维度
    #density = 0.9  #train test 比
    c = 1    #柯西估计器参数
    lamdau = 1  #正则项
    lamdas = 1  #正则项
    beta = 0.003  #学习率       #best 0.001附近
    iteration = 1000 #最大迭代次数
    decay_steps = 50   #用来控制衰减速度
    decay_rate = 0.9   #指数衰减学习率  
    
    if dataset =='tp':
        filedir = "dataset1/tpMatrix.txt"
        outlier_filename = "outlier_tp/Full_I_outlier_fra_"+str(outlier_fra) + ".npy"
    if dataset =='rt':
        filedir = "dataset1/rtMatrix.txt"
        outlier_filename = "outlier_rt/Full_I_outlier_fra_"+str(outlier_fra) + ".npy"
    
    R = load_data(filedir)
    I_outlier = np.load(outlier_filename)
    I_train,I_test = train_test(R,density)
    I_test = merge_test_outlier(I_outlier,I_test)
    
   
    U,S, CMF_loss, CMF_rmse, CMF_mae = CMF(R,l,c,lamdau,lamdas,beta,iteration,\
                                      decay_steps,decay_rate, I_train,I_test)

    
    print ("%.4f,%.4f" % (min(CMF_rmse), min(CMF_mae)))
    print ("%.4f,%.4f" % (CMF_rmse[-1], CMF_mae[-1]))
    
    
import sys
if __name__ == '__main__':
    density = float(sys.argv[1])
    outlier_fra = float(sys.argv[2])
    dataset = sys.argv[3]
    main(density, outlier_fra, dataset)
