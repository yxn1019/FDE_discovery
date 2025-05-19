import numpy as np
import torch
from torch.nn import Linear,Tanh,Sequential
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import torch.nn.functional as func
import os
from matplotlib import cm
import scipy.io as io
#%% -------------------set printoptions-------------------------
torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)



#------------------Neural network setting-----------------
#Loss_function
class PINNLossFunc(nn.Module):
    def __init__(self,h_data):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return

    def forward(self,prediction):
        f1=torch.pow((prediction-self.h_data),2).sum()
        MSE=f1/total
        return MSE
class SinActivation(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)
class GaussianActivation(torch.nn.Module):
    def forward(self, x):
        return torch.exp(-x ** 2)
#Neural Network
Net=Sequential(
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


#%%#############################-------dataset and parameters
soil = 'silt'
data = io.loadmat('dataset/silt.mat')
un = np.real(data['ex'])
# x = np.real(data['x'][0])
t = np.real(data['t'])
# x=torch.arange(0,30,0.25)
# t=torch.arange(0,15,0.1)
noise_level = 0  # percentage of noise
x_num=1
t_num=len(t)
total=x_num*t_num   #Num of total data
choose=15   #Num of training data
choose_validate=t_num-choose  #Num of validate data   
######################################################
# Optimizer
optimizer=torch.optim.Adam([
    {'params': Net.parameters()}
    #{'params': theta},
], lr=1e-3, weight_decay=1e-4)

#---------------Create Folder----------------------
try:
    os.makedirs('random_ab')
except OSError:
    pass

try:
    os.makedirs('model_save/%s-%d-%d'%(soil,choose,noise_level))
except OSError:
    pass

#%-----------------Preparing Training and Validate Dataset
un_raw=torch.from_numpy(un.astype(np.float32))
data=torch.zeros(1)
h_data=torch.zeros([total,1])
database=torch.zeros([total,1])
num=0

    # for j in range(x_num):
        # data[0]=x[j]
data=torch.tensor(t,requires_grad=True).float()
h_data=torch.tensor(un)  #Add noise
database=data


#-----------Randomly choose----------------
a = random.sample(list(range(0,20))+list(range(30, total)), choose)
# a = random.sample(range(0, total-1), choose)
np.save("random_ab/"+"a-%d.npy"%(choose),a)
temp=[]
for i in range(total):
    if i not in a:
        temp.append(i)
b=random.sample(temp, choose_validate)

h_data_choose = torch.zeros([choose, 1])
database_choose = torch.zeros([choose, 1])
h_data_validate= torch.zeros([choose_validate, 1])
database_validate = torch.zeros([choose_validate, 1])
num = 0
# for i in a:
#     h_data_choose[num] = h_data[i]
#     database_choose[num] = database[i]
#     num += 1
# num=0
h_data_choose = h_data[a]
database_choose = database[a]
# for i in b:
#     h_data_validate[num] = h_data[i]
#     database_validate[num] = database[i]
#     num += 1
h_data_validate = h_data[b]
database_validate = database[b]

# Max_iter_num=2000000
Max_iter_num=10001
torch.manual_seed(525)
with open('model_save/%s-%d-%d/'%(soil,choose,noise_level)+'data.txt', 'w') as f:  # 设置文件对象
    for i in range(Max_iter_num):
        optimizer.zero_grad()
        prediction = Net(database_choose)
        prediction_validate = Net(database_validate).cpu().data.numpy()
        a = PINNLossFunc(h_data_choose)
        loss = a(prediction)
        loss_validate = np.sum((h_data_validate.data.numpy() - prediction_validate) ** 2) / choose_validate
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 1000 == 0:
            print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (i, loss, loss_validate))
            f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (i, loss, loss_validate))
            if int(i / 100) == 800:
                # sign=stop(loss_validate_record)
                # if sign==0:
                #     break
                break
            if i>1000:
                torch.save(Net.state_dict(), 'model_save/%s-%d-%d/'%(soil,choose,noise_level)+"%s-%d.pkl"%(soil,i))

#%%
# t = database_validate[:,1]
# t = t.data.numpy()
# x = x.data.numpy()
c = Net(database).data.numpy().reshape(t_num,x_num)
un_noise = h_data.data.numpy().reshape(t_num,x_num)
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# X, T = np.meshgrid(x,t) #mesh for train
# ax.plot_surface(T, X, c, cmap='viridis') #NN模拟值
# ax.scatter(T, X, un_noise,               #加噪精确解
#             facecolors = 'none', 
#             marker = '*', 
#             edgecolor = 'k', 
#             s = 30,
#             label = 'Exact')
# ax.set_xlabel(r'$T$')
# ax.set_ylabel(r'$X$')
# ax.set_zlabel(r'$u$')
plt.plot(t,c)
plt.plot(t,un,'x')
plt.show()