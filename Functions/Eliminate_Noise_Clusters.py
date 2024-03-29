import numpy as np

from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation

def Eliminate_Noise_Clusters(Nc):
    global KM
    Xrj = np.empty((0,KM[0]["data"].shape[1]))
    Xcl = np.empty((0,KM[0]["data"].shape[1]))
    m = 0

    while m < len(KM):
        
        if KM[m]["data"].shape[0] < Nc :
            Xrj = np.append(Xrj,KM[m]["data"])
            # KM[m] = []
            KM = np.delete(KM,m)
            
        else :
            Xcl = np.append(Xcl,KM[m]["data"],axis=0)

        m = m + 1
    # print(KM)
    return Xcl,Xrj





# import pandas as pd

# # data = pd.read_csv('Env_nonstat.csv', sep='\t')

# # xo = np.array(data)[1:2595,0]
# # X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X) 
# gamma = 2
# nu = 0.3
# eta = 0.2
# X0 = X[0,:].reshape(1,-1)
# Xnew = X[1,:].reshape(1,-1)
# KM = Kernel_Machine_Initialisation(X0, gamma, nu, eta,[])
# # KM = Kernel_Machine_Initialisation(Xnew, gamma, nu, eta,KM1)
# Nc = 10
# e = Eliminate_Noise_Clusters(Nc)
# print(e)