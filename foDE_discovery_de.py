# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:04:54 2024

@author: 54543
"""
# locals().clear()
import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch.nn.functional as func
import scipy.special as sp
# from STRidge import *
from scipy.optimize import differential_evolution as de
from scipy.optimize import minimize
from foSTRidge import *
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


#%%
# Optimizer

# 目标函数，用于计算模拟数据与实际数据的误差
def objective_function(params):
    global iter
    # noise_level = 0
    alpha = params
    _,err = foSTRidge(alpha)
    # 计算最后时刻的模拟数据与实际数据的error
    print(f'Iteration {iter}: loss = {err}, alpha = {alpha}')
    iter = iter + 1
    return err

# 设置一些示例数据
np.random.seed(42)

# Initial guess for parameters
iter = 1
bounds=[(0.01, 0.99)]
alpha0 = np.random.rand(1)
# 全局优化
# result = de(objective_function
#             ,bounds
#             # ,maxiter=5 # (maxiter+1)*num_pop*num_para, if maxiter=2, then times=60
#             # ,tol=1e-4
#             # , disp=True
#             )
# 局部优化
result = minimize(objective_function, alpha0, method='Powell', bounds=[(0.01, 0.99)])
alpha_opt = result.x
w,_ = foSTRidge(alpha_opt)
#%%
print(w)
print(f"best parameters: alpha = {alpha_opt}")
print(f'eta={1.11/w[0,0]}, E={-1.11/w[0,0]*w[1,0]}')
print(f'best loss: {result.fun}')
# 方程参数


