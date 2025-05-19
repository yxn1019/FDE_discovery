import numpy as np
import random
#---------------------------------一些函数--------------------------------------
def STRidge(X0,y,lam,maxit,tol,normalize=2,print_results=False):
    #tol is threhold
    n,d=X0.shape
    X=np.zeros((n,d))
    #Normalization
    if normalize!=0:
        Mreg=np.zeros((d,1))
        for i in range(0,d):
            Mreg[i]=1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i]=Mreg[i]*X0[:,i]
    else:X=X0

    #Estimate of Ridge
    if lam!=0:
        w=np.linalg.lstsq(X.T.dot(X)+lam*np.eye(d),X.T.dot(y))[0]
        # print(w)
    else:
        w=np.linalg.lstsq(X,y)[0]
    num_relevant=d
    biginds=np.where(abs(w)>tol)[0]

    for j in range(maxit):
        #Find term to be deleted
        smallinds=np.where(abs(w)<tol)[0]
        new_biginds=[i for i in range(d) if i not in smallinds]

        #stop if no change happens
        if num_relevant==len(new_biginds):
            break
        else:
            num_relevant=len(new_biginds)

        #garantee no null coeffcient
        if len(new_biginds)==0:
            if j==0:
                #if print_results:print"Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds=new_biginds

        w[smallinds]=0
        if lam!=0:
            w[biginds]=np.linalg.lstsq(X[:,biginds].T.dot(X[:,biginds])+lam*np.eye(len(biginds)),X[:,biginds].T.dot(y))[0]
        else:
            w[biginds]=np.linalg.lstsq(X[:,biginds],y)[0]
    # print(np.linalg.lstsq(X[:,biginds],y)[0])
        #get w
    if len(biginds)!=[]:
        w[biginds]=np.linalg.lstsq(X[:,biginds],y)[0]

    if normalize!=0:
        return np.multiply(Mreg,w)
    else:
        return w


# R=np.load("Theta-1%.npy")
# # for i in range(1,10):
# #     c=(R[:,i]-np.mean(R[:,i]))/np.std(R[:,i])
# #     R[:,i]=c
# Ut=np.load("Ht_n-1%.npy")
# # c=(Ut-np.mean(Ut))/np.std(Ut)
