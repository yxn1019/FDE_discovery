# locals().clear()
import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import torch.nn as nn
import random
# import torch.nn.functional as func
import scipy.special as sp
from STRidge import *
#%%
def foSTRidge(alpha, noise_level=0):
    def fractional_integral(f, alpha, w, b):
        # x, w = sp.roots_jacobi(n, 0, 1-alpha) 
        # t = b-b/2*(x+1) # transform variable
        # Compute the Caputo fractional derivative using the quadrature approximation
        # f: Function for which is computed by the Caputo fractional derivative
        # alpha: Fractional integral order, the derivative order is n-alpha 
        # b: integrand point/integral upper limit
        # n: Number of quadrature points
        # Calculate the Jacobi-Gauss quadrature points and weights in the interval (0, b)
        if alpha>1.:
            result = np.matmul(f,w.reshape(-1,1))/sp.gamma(2-alpha)*(b/2)**(2-alpha)
        else:# i.e. 0<alpha<1
            # x, w = sp.roots_jacobi(n, 0, -alpha) 
            # t = b-b/2*(x+1) # transform variable
            result = np.matmul(f,w.reshape(-1,1))/sp.gamma(1-alpha)*(b/2)**(1-alpha)   
        # print('caputo=',result)
        return result
    class SinActivation(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)
    class GaussianActivation(torch.nn.Module):
        def forward(self, x):
            return torch.exp(-x ** 2)
    #Neural Network
    Net = Sequential(
        Linear(1, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 20),
        # SinActivation(),
        Tanh(),
        Linear(20, 1),
    )

    #Generating meta-data
    n = 5 # Number of quadrature points
    alpha = np.random.rand(1) # time fractional order
    # beta = 1.7 # space fractional order

    num=0
    Net.load_state_dict(torch.load('model_save/clay-15-0/clay-10000.pkl'))
    # x=torch.arange(5, 25, 0.1)
    t=torch.arange(0.02, 8, 0.02)

    x_num=1
    t_num=len(t)
    total = x_num*t_num

    if 0<alpha<1:
        t_gj, w_gj = sp.roots_jacobi(n, 0, -alpha) # G-J points, 0<alpha<1
    # if 1<beta<2:
    #     x_gj, v_gj = sp.roots_jacobi(n, 0, 1-beta) # G-J points, 1<alpha<2

    data = torch.zeros(1)
    h_data = torch.zeros([total,1])
    database = torch.zeros([total,1])
    database_tf = torch.zeros([total*n,1])
    # database_sf = torch.zeros([total*n,2])

    database = t.reshape(-1,1)

    for i in range(total):
        database_tf[i*n:i*n+n] = database[i]-database[i]/2*(t_gj.reshape(-1,1)+1)

     
        
    #-----Automatic differentiation--------
    # print('Building the library...')
    database = Variable(database, requires_grad=True)
    database_n = database.data.numpy()
    database_tf = Variable(database_tf, requires_grad=True)
    # database_sf = Variable(database_sf, requires_grad=True)
    PINNstatic = Net(database)
    PINNstatic_tf = Net(database_tf)
    # PINNstatic_sf = Net(database_sf)
    H = PINNstatic.data.numpy()
    H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
    H_grad_tf = torch.autograd.grad(outputs=PINNstatic_tf.sum(), inputs=database_tf, create_graph=True)[0]
    # H_grad_sf = torch.autograd.grad(outputs=PINNstatic_sf.sum(), inputs=database_sf, create_graph=True)[0]
    #print(H_grad)
    # Hx=H_grad[:,0].reshape(total,1)
    # Hx_n=Hx.data.numpy()
    #print(Hx)
    Ht=H_grad.reshape(total,1)
    Ht_n=Ht.data.numpy()
    # time fractional derivative
    Ht_f=H_grad_tf.reshape(-1,n)
    Ht_f_n=Ht_f.data.numpy()
    Halpha_n = np.zeros([total,1])
    Halpha_n = fractional_integral(Ht_f_n, alpha, w_gj, database_n.reshape(-1,1))
    HH = PINNstatic * PINNstatic
    HH_n = HH.data.numpy()
    Htt=torch.autograd.grad(outputs=Ht.sum(), inputs=database,create_graph=True)[0].reshape(total,1)
    Htt_n=Htt.data.numpy()
    HHt=PINNstatic * Ht
    HHt_n=HHt.data.numpy()
    HtHt = Ht*Ht
    HtHt_n = HtHt.data.numpy()
    HHtt = PINNstatic * Htt
    HHtt_n = HHtt.data.numpy()
    HtHtt = Ht*Htt
    HtHtt_n = HtHtt.data.numpy()
    HttHtt = Htt*Htt
    HttHtt_n = HttHtt.data.numpy()

    #------------- Building library --------------

    #Theta=Hx_n
    Theta = np.ones([total,1])
    Theta = np.hstack((Theta, H))
    Theta = np.hstack((Theta, HH_n))
    Theta = np.hstack((Theta, Ht_n))
    Theta = np.hstack((Theta, HHt_n))    
    Theta = np.hstack((Theta, Htt_n))
    Theta = np.hstack((Theta, HtHt_n))
    Theta = np.hstack((Theta, HHtt_n))
    Theta = np.hstack((Theta, HtHtt_n))
    Theta = np.hstack((Theta, HttHtt_n))


    # np.save("Theta-1%",Theta)
    # np.save("Ht_n-1%",Halpha_n)

    # print(Theta)
    # print(Ht_n)
    #%% --------------PDE discovery-------------------
    # print('Starting Discovery...')
    R = Theta
    Ut = Halpha_n
    # Ut = Ht_n
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
        l0_penalty = 1
    #用最小二乘估计
    w=np.zeros((D,1))
    w_best=np.linalg.lstsq(TrainR,TrainY)[0]
    err_best=np.linalg.norm(TestY-TestR.dot(w_best),2)+l0_penalty*np.count_nonzero(w_best)
    # k = np.count_nonzero(w_best) + 1
    # err_best = -0.5 * D * (np.log(2 * np.pi * err_best) + 1)    #likelihood
    # err_best = 2 * k - 2 * err_best                             #aic
    tol_best=0.1

    #提高超参数值直至测试表现下降
    for iter in range(maxit):
        #获得一系列参数和误差
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize=normalize)
        err = np.linalg.norm(TestY-TestR.dot(w),2)+l0_penalty*np.count_nonzero(w)
        # k = np.count_nonzero(w) + 1
        # err = -0.5 * D * (np.log(2 * np.pi * err) + 1)      #likelihood
        # err = 2 * k - 2 * err                               #aic
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
        # print(w.T)
    return w_best, err_best
#%% print weights and MSE
# w_best, err_best = fSTRidge(0.9,1.8)
# print(w_best)   #Discovered terms and coefficients
# print('L0_MSE=', err_best)   #MSE

