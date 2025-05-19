import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch.nn.functional as func
from STRidge import *

class LearnableGaussianActivation(torch.nn.Module):
    def __init__(self):
        super(LearnableGaussianActivation, self).__init__()
        # Initialize learnable parameters mu and sigma
        self.mu = torch.nn.Parameter(torch.randn(1))  # Mean parameter
        self.sigma = torch.nn.Parameter(torch.ones(1))  # Standard deviation parameter
    def forward(self, x):
        # Gaussian function: exp(-(x - mu)^2 / (2 * sigma^2))
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
#Neural Network
Net = Sequential(
    Linear(2, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 20),
    Tanh(),
    Linear(20, 1),
)
# Net = Sequential(
#     Linear(2, 20),
#     LearnableGaussianActivation(),  # Custom Gaussian activation
#     Linear(20, 20),
#     LearnableGaussianActivation(),
#     Linear(20, 20),
#     LearnableGaussianActivation(),
#     Linear(20, 20),
#     LearnableGaussianActivation(),
#     Linear(20, 20),
#     LearnableGaussianActivation(),
#     Linear(20, 1)
# )

#Generating meta-data
Net.load_state_dict(torch.load('model_save/ade-4000-25/ade-5000.pkl'))
x=torch.arange(4, 26, 0.1)
t=torch.arange(3, 14, 0.1)
x_num=len(x)
t_num=len(t)
total=x_num*t_num
num=0
data=torch.zeros(2)
h_data=torch.zeros([total,1])
database=torch.zeros([total,2])
for i in range(t_num):
    for j in range(x_num):
        data[0]=x[j]
        data[1]=t[i]
        database[num]=data
        num+=1

#Automatic differentiation
database = Variable(database, requires_grad=True)
PINNstatic=Net(database)
# for i in range(1010):
#     print(PINNstatic[i])
H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
#print(H_grad)
Hx=H_grad[:,0].reshape(total,1)
Hx_n=Hx.data.numpy()
#print(Hx)
Ht=H_grad[:,1].reshape(total,1)
Ht_n=Ht.data.numpy()
#print(Ht)
Hxx=torch.autograd.grad(outputs=Hx.sum(), inputs=database,create_graph=True)[0][:,0].reshape(total,1)
Hxx_n=Hxx.data.numpy()
Hxxx=torch.autograd.grad(outputs=Hxx.sum(), inputs=database,create_graph=True)[0][:,0].reshape(total,1)
Hxxx_n=Hxxx.data.numpy()
Htt=torch.autograd.grad(outputs=Ht.sum(), inputs=database,create_graph=True)[0][:,1].reshape(total,1)
Htt_n=Htt.data.numpy()
Hxt=torch.autograd.grad(outputs=Ht.sum(), inputs=database,create_graph=True)[0][:,0].reshape(total,1)
HHxx = PINNstatic* Hxx
HHxx_n=HHxx.data.numpy()
HHx = PINNstatic * Hx
HHx_n=HHx.data.numpy()
HHxxx = PINNstatic * Hxxx
HHxxx_n=HHxxx.data.numpy()
HHHxx = PINNstatic* HHxx
HHHxx_n=HHHxx.data.numpy()
HHHx = PINNstatic * HHx
HHHx_n=HHHx.data.numpy()
HHHxxx = PINNstatic * HHxxx
HHHxxx_n=HHHxxx.data.numpy()


#Theta=Hx_n
Theta=np.ones([total,1])
Theta = np.hstack((Theta, Hx_n))
Theta = np.hstack((Theta, Hxx_n))
Theta=np.hstack((Theta, Hxxx_n))
Theta = np.hstack((Theta, HHx_n))
Theta = np.hstack((Theta, HHxx_n))
Theta=np.hstack((Theta,HHxxx_n))
Theta=np.hstack((Theta,HHHx_n))
Theta=np.hstack((Theta,HHHxx_n))
Theta=np.hstack((Theta,HHHxxx_n))
np.save("Theta-1%",Theta)
np.save("Ht_n-1%",Ht_n)

#%% -----------discovery------------
# print(Theta)
# print(Ht_n)
R = Theta
# Ut = Halpha_n
Ut = Ht_n
# c=(Ut-np.mean(Ut))/np.std(Ut)
# Ut=c
lam=2
d_tol=0.1
maxit = 25
STR_iters = 10
l0_penalty = None
normalize = 0
split = 0.8
print_best_tol = True
#np.random.seed(0)
nn,_=R.shape
train=np.random.choice(nn,int(nn*split),replace=False)
test=[i for i in np.arange(nn) if i not in train]
TrainR=R[train,:]
TestR=R[test,:]
TrainY=Ut[train,:]
TestY=Ut[test,:]
D=TrainR.shape[1]
#设置最初的超参数和l0惩罚项
d_tol=float(d_tol)
tol=d_tol
if l0_penalty==None:
##############-------------l0
    l0_penalty=0.5
##############---------------
#用最小二乘估计
w=np.zeros((D,1))
w_best=np.linalg.lstsq(TrainR,TrainY)[0]
err_best=np.linalg.norm(TestY-TestR.dot(w_best),2)+l0_penalty*np.count_nonzero(w_best)
# k = np.count_nonzero(w_best) + 2                # 2 more fractional order
# err_best = -0.5 * D * (np.log(2 * np.pi * err_best) + 1)    #likelihood
# err_best = 2 * k - 2 * err_best   
tol_best=0.1

#提高超参数值直至测试表现下降
for iter in range(maxit):
    #获得一系列参数和误差
    w = STRidge(R,Ut,lam,STR_iters,tol,normalize=normalize)
    err = np.linalg.norm(TestY-TestR.dot(w),2)+l0_penalty*np.count_nonzero(w)
    # k = np.count_nonzero(w) + 2
    # err = -0.5 * D * (np.log(2 * np.pi * err) + 1)    #likelihood
    # err = 2 * k - 2 * err                             #aic
    #判断精度有没有上升
    if err<=err_best:
        err_best=err
        w_best=w
        tol_best=tol
        tol=tol+d_tol

    else:
        tol=max([0,tol-2*d_tol])
        d_tol=2*d_tol/(maxit-iter)
        tol=tol+d_tol

# if print_best_tol:
#     print("optimal tolerance:",tol_best)
print('error=',err_best,'\n','w=\n',w)