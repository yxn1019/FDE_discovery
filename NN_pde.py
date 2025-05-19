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

#---------------Parameter setting--------------
#%%-----------------------data and parameters--------
# folderpath = "dataset/"
datafile = r"dataset/Convection_diffusion.dat"
# filename = r"dataset/tsfade_fft.dat"
# filepath = os.path.join(folderpath, filename)
pos = []
j=0
unm=0
model_file = 'ade' # 'ade' or 'fde'
activation = 'tanh' # 'tanh' or 'gauss'
Max_iter_num = 10001
x=torch.arange(0,30,0.25)
t=torch.arange(0,15,0.1)
noise_level = 25  # percentage of noise
x_num=len(x)
t_num=len(t)
total=x_num*t_num   #Num of total data
choose=4000   #Num of training data
choose_validate=4000  #Num of validate data
total=x_num*t_num   #Num of total data
un=np.ones([t_num,x_num])
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
class LearnableGaussianActivation(nn.Module):
    def __init__(self):
        super(LearnableGaussianActivation, self).__init__()
        # Initialize learnable parameters mu and sigma
        self.mu = nn.Parameter(torch.randn(1))  # Mean parameter
        self.sigma = nn.Parameter(torch.ones(1))  # Standard deviation parameter

    def forward(self, x):
        # Gaussian function: exp(-(x - mu)^2 / (2 * sigma^2))
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))
#Neural Network

if activation == 'gauss':
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
else:
    Net=Sequential(
        Linear(2,20),
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

    
#Optimizer
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
    os.makedirs('model_save/%s-%d-%d'%(model_file,choose,noise_level))
except OSError:
    pass


#################################################
with open(datafile, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        for line in lines.split():
            pos.append(line)
        for i in range(len(pos)):
            un[j,i]=float(pos[i])
        pos=[]
        j+=1
        unm+=1
        if not lines:
            break
        
# data = io.loadmat('dataset/tsfade_fft.mat')
# un = np.real(data['Exact'])
# x = np.real(data['x'][0])
# t = np.real(data['t'][0])

#-----------------Preparing Training and Validate Dataset
un_raw=torch.from_numpy(un.astype(np.float32))
data=torch.zeros(2)
h_data=torch.zeros([total,1])
database=torch.zeros([total,2])
num=0
for i in range(t_num):
    for j in range(x_num):
        data[0]=x[j]
        data[1]=t[i]
        h_data[num]=un[i,j]*(1+0.01*noise_level*np.random.uniform(-1,1))  #Add noise
        database[num]=data
        num+=1

#-----------Randomly choose----------------
a = random.sample(range(0, total-1), choose)
# a = random.sample(list(range(0,1200))+list(range(2000, total)), choose)
np.save("random_ab/"+"a-%d.npy"%(choose),a)
temp=[]
for i in range(total):
    if i not in a:
        temp.append(i)
b=random.sample(temp, choose_validate)

h_data_choose = torch.zeros([choose, 1])
database_choose = torch.zeros([choose, 2])
h_data_validate= torch.zeros([choose_validate, 1])
database_validate = torch.zeros([choose_validate, 2])
num = 0
for i in a:
    h_data_choose[num] = h_data[i]
    database_choose[num] = database[i]
    num += 1
num=0
for i in b:
    h_data_validate[num] = h_data[i]
    database_validate[num] = database[i]
    num += 1


# Max_iter_num=2000000

torch.manual_seed(525)
with open('model_save/%s-%d-%d/'%(model_file,choose,noise_level)+'data.txt', 'w') as f:  # 设置文件对象
    for i in range(Max_iter_num):
        optimizer.zero_grad()
        prediction = Net(database_choose)
        prediction_validate = Net(database_validate).cpu().data.numpy()
        a = PINNLossFunc(h_data_choose)
        loss = a(prediction)
        loss_validate = np.sum((h_data_validate.data.numpy() - prediction_validate) ** 2) / choose_validate
        loss.backward()
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
                torch.save(Net.state_dict(), 'model_save/%s-%d-%d/'%(model_file,choose,noise_level)+"%s-%d.pkl"%(model_file,i))

#%%
# t = database_validate[:,1]
# t = t.data.numpy()
# x = x.data.numpy()
c = Net(database).data.numpy().reshape(t_num,x_num)
un_noise = h_data.data.numpy().reshape(t_num,x_num)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, T = np.meshgrid(x,t) #mesh for train
ax.plot_surface(T, X, c, cmap='viridis') #NN模拟值
ax.scatter(T, X, un_noise,               #加噪精确解
            facecolors = 'none', 
            marker = '*', 
            edgecolor = 'k', 
            s = 30,
            label = 'Exact')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$X$')

ax.set_zlabel(r'$u$')

# ax.set_title('Original Sharp Data')
# Plot the interpolated data
# ax = fig.add_subplot(122, projection='3d')
# T_interp, X_interp = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
# ax.plot_surface(T_interp, X_interp, un, cmap='viridis')
# ax.set_title('DNN Interpolated Data')
plt.show()