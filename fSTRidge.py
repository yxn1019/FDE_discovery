# locals().clear()
import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
# import torch.nn.functional as F
from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import torch.nn as nn
import random
# import torch.nn.functional as func
import scipy.special as sp
from STRidge import *
#%%
def fSTRidge(alpha, beta, noise_level=0, lamb=0.001,file_read=''
             , model_file = 'ade', activation = 'gauss'):
    def fractional_integral(f, alpha, w, b):
        # f: Function for which is computed by the Caputo fractional derivative
        # alpha: Fractional integral order, the derivative order is n-alpha 
        # a, b: Interval for the integral approximation, a=0
        # n: Number of quadrature points
        # Calculate the Jacobi-Gauss quadrature points and weights in the interval (a, b)
        if alpha>1.:
            # x, w = sp.roots_jacobi(n, 0, 1-alpha) 
            # t = b-b/2*(x+1) # transform variable
            # Compute the Caputo fractional derivative using the quadrature approximation
            result = np.matmul(f,w.reshape(-1,1))/sp.gamma(2-alpha)*(b/2)**(2-alpha)
        else:# i.e. 0<alpha<1
            # x, w = sp.roots_jacobi(n, 0, -alpha) 
            # t = b-b/2*(x+1) # transform variable
            result = np.matmul(f,w.reshape(-1,1))/sp.gamma(1-alpha)*(b/2)**(1-alpha)   
        # print('caputo=',result)
        return result
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
  
    #Generating meta-data
    n = 5 # Number of quadrature points
    # alpha = 0.9 # time fractional order
    # beta = 1.8 # space fractional order

    if activation == 'tanh':
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
    else:
        Net = Sequential(
            Linear(2, 20),
            LearnableGaussianActivation(),  # Custom Gaussian activation
            Linear(20, 20),
            LearnableGaussianActivation(),
            Linear(20, 20),
            LearnableGaussianActivation(),
            Linear(20, 20),
            LearnableGaussianActivation(),
            Linear(20, 20),
            LearnableGaussianActivation(),
            Linear(20, 1)
           )
    num=0
    # print(file_read+'/'+model_file+'-5000.pkl')
    Net.load_state_dict(torch.load(file_read+'/'+model_file+'-5000.pkl'))
    x=torch.arange(4, 26, 0.1)
    t=torch.arange(3, 14, 0.1)
    
    x_num=len(x)
    t_num=len(t)
    total = x_num*t_num
    t_n = t.data.numpy()
    x_n = x.data.numpy()
    if alpha<1:
        t_gj, w_gj = sp.roots_jacobi(n, 0, -alpha) # G-J points, 0<alpha<1
    else:
        t_gj, w_gj = sp.roots_jacobi(n, 0, 1-alpha) # G-J points, 1<alpha<2
    if beta >1:
        x_gj, v_gj = sp.roots_jacobi(n, 0, 1-beta) # G-J points, 1<alpha<2
    else:
        x_gj, v_gj = sp.roots_jacobi(n, 0, -beta) # G-J points, 0<alpha<1
    
    data = torch.zeros(2)
    h_data = torch.zeros([total,1])
    database = torch.zeros([total,2])
    database_tf = torch.zeros([total*n,2])
    database_sf = torch.zeros([total*n,2])
    
    for i in range(t_num):
        for j in range(x_num):
            data[0]=x[j]
            data[1]=t[i]
            database[num]=data
            num+=1
    
    for i in range(total): #坐标变换，为求G-J积分
        database_tf[i*n:i*n+n,1] = database[i,1]-database[i,1]/2*(t_gj+1)
        database_tf[i*n:i*n+n,0] = database[i,0]
        database_sf[i*n:i*n+n,0] = database[i,0]-database[i,0]/2*(x_gj+1)
        database_sf[i*n:i*n+n,1] = database[i,1]
        
            
    #-----Automatic differentiation--------
    # print('Building the library...')
    database = Variable(database, requires_grad=True)
    database_n = database.data.numpy()
    database_tf = Variable(database_tf, requires_grad=True)
    database_sf = Variable(database_sf, requires_grad=True)
    PINNstatic = Net(database)
    PINNstatic_tf = Net(database_tf)
    PINNstatic_sf = Net(database_sf)
    H = PINNstatic
    H_n = H.data.numpy()
    HH = H*H
    HH_n = HH.data.numpy()
    H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
    H_grad_tf = torch.autograd.grad(outputs=PINNstatic_tf.sum(), inputs=database_tf, create_graph=True)[0]
    H_grad_sf = torch.autograd.grad(outputs=PINNstatic_sf.sum(), inputs=database_sf, create_graph=True)[0]
    #print(H_grad)
    Hx=H_grad[:,0].reshape(total,1)
    Hx_n=Hx.data.numpy()
    #print(Hx)
    Ht=H_grad[:,1].reshape(total,1)
    Ht_n=Ht.data.numpy()
    # time fractional derivative
    Ht_f=H_grad_tf[:,1].reshape(-1,n)
    Ht_f_n=Ht_f.data.numpy()
    Halpha_n = np.zeros([total,1])
    Halpha_n = fractional_integral(Ht_f_n, alpha, w_gj, database_n[:,1].reshape(-1,1))
    
    Hxx=torch.autograd.grad(outputs=Hx.sum(), inputs=database,create_graph=True)[0][:,0].reshape(total,1)
    Hxx_n=Hxx.data.numpy()
    
    # space fractional derivative
    Hbeta_n = np.zeros([total,1])
    Hx_f=H_grad_sf[:,0]
    Hxx_f=torch.autograd.grad(outputs=Hx_f.sum(), inputs=database_sf,create_graph=True)[0][:,0].reshape(-1,n)
    Hxx_f_n=Hxx_f.data.numpy()
    # for i in range(total):
    Hbeta_n = fractional_integral(Hxx_f_n, beta, v_gj, database_n[:,0].reshape(-1,1))
        
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
    
    
    #------------- Building library --------------
    
    #Theta=Hx_n
    Theta = np.ones([total,1])
    Theta = np.hstack((Theta, H_n))
    # Theta = np.hstack((Theta, Halpha_n))
    Theta = np.hstack((Theta, Hx_n))
    Theta = np.hstack((Theta, Hbeta_n))
    # Theta = np.hstack((Theta, Hxx_n))
    Theta = np.hstack((Theta, Hxxx_n))
    Theta = np.hstack((Theta, HH_n))
    Theta = np.hstack((Theta, HHx_n))
    Theta = np.hstack((Theta, HHxx_n))
    Theta = np.hstack((Theta, HHxxx_n))
    Theta = np.hstack((Theta, HHHx_n))
    Theta = np.hstack((Theta, HHHxx_n))
    Theta = np.hstack((Theta, HHHxxx_n))
    
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
##############-------------l0
        l0_penalty=lamb*np.linalg.cond(R)
        # print(lamb, '\n', l0_penalty)
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
    # print('error=',err_best)
    return w_best, err_best
#%% print weights and MSE
# w_best, err_best = fSTRidge(0.9,1.8)
# print(w_best)   #Discovered terms and coefficients
# print('L0_MSE=', err_best)   #MSE

