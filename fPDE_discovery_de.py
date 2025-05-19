# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:04:54 2024

@author: 54543
"""
# locals().clear()
import numpy as np
# import torch
# from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import random
# import torch.nn.functional as func
# import scipy.special as sp
# from STRidge import *
from scipy.optimize import differential_evolution as de
from scipy.optimize import minimize
from fSTRidge import *
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import os

np.set_printoptions(linewidth=500)
#%%
# Optimizer

# 目标函数
def objective_function(params):
    global iter
    alpha, beta = params
    w,err = fSTRidge(alpha, beta, noise_level=noise_level,lamb =lamb, file_read=file_read, model_file = model_file, activation = activation)
    # 计算最后时刻的模拟数据与实际数据的err
    # np.savetxt('parameter.txt', np.array2string(w.T), delimiter='\n')
    print(f'Iteration {iter}: loss = {err}, alpha = {alpha:.6f}, beta = {beta:.6f}\t\n'+np.array2string(w.T))
    print(f'{iter} \t {err:.6f} \t {alpha:.4f} \t {beta:.4f}\t'+np.array2string(w.T.flatten()).strip('[[').strip(']]'),file=file)
    # if iter%10==0:
    # print(w.T)
    iter = iter+1
    return err

np.random.seed(42)# 设置一些示例数据

################### main
# Initial parameters
noise_level = 25
lamb = 0.001
iter = 1
trained_point = 4000 # number of selected point by DNN
bounds=[(0.01, 0.99999),(1.01, 1.99999)]
# 优化
model_file = 'ade'
file_read = f'model_save/{model_file}-{trained_point}-{noise_level}'
activation = 'tanh'
try:
    os.makedirs(file_read)
except OSError:
    pass
# file = open('parameter.txt','w')
with open(file_read+'/parameter.txt', 'w') as file:  # 设置文件对象
    coeff = de(objective_function
                ,bounds
                ,maxiter=15 # (maxiter+1)*num_pop*num_para, if maxiter=2, then max times=60
                ,tol=1e-4
                , disp=True
                )
    
# 优化结果
# coeff = minimize(objective_function, coeff.x, method='Powell', bounds=[(0.01, 0.99), (1.01, 1.99)])
alpha_opt, beta_opt = coeff.x
w,_ = fSTRidge(alpha_opt, beta_opt,noise_level=noise_level, lamb = lamb, file_read=file_read, model_file = model_file, activation = activation)
#%%
print(f"best parameters: alpha = {alpha_opt:.5f}, beta = {beta_opt:.5f}")
print(f'best loss: {coeff.fun}')
# 方程参数
print(w)
file.close()
