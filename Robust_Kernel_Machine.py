import numpy as np
import pandas as pd

from KernelUpdate_NormTruncSGradient import KernelUpdate_NormTruncSGradient
#from Kernel_Merging_Process import Kernel_Merging_Process
from Kernel_Merging_Function import Kernel_Merging_Function



def Robust_Kernel_Machine(Process,WK_Sim,Xnew,gamma,nu,eta,kard,KM):
# >>INPUT:
#     Process: etat 0 si la donnée est traité 1 si Ambiguité (Mergin)
#     WK_Sim : Noyau gagnant et Valeur de Similarity correspondante avec l'observation XT
#     Xnew : nouvelle Observation acquise à l'instant 
#     gamma: Parametre du RBF, correspond à 1/2v² où v² est la variance des données
#     nu: Paramètre du Slack-error (admise), Règlage du nombre de SV et non-SV
# >>OUTPUT:

    #global KM
    #global X
    # KM = []
    if Process == 1 :
        return
    
    
    
    nkern = len(WK_Sim)
    idkern = []
    
    #print(WK_Sim)
    for v in WK_Sim :
        
        id = v[0]
        idkern.append(id)
    
    Wnmk = np.sort(idkern)
    
    
    KERN = []
    
    for id in range(3) :
        KERN.append(KM[id])
    
    # KernMerg = Kernel_Merging_Function(KM[Wnmk],nkern, Wnmk[0],gamma,nu,eta,kard,KM)
 
    KernMergUpdat = KernelUpdate_NormTruncSGradient(WK_Sim[0],Xnew,gamma,nu,eta,kard,KM)

    #------------------------ Mis à jour du noyau----------------------------------#

    KM = np.resize(KM, len(KM) -nkern + 1) #?????????????????????
    #print(len(KM))
    KM[Wnmk[0]] = KernMergUpdat
    KM = np.resize(KM, len(KM) -len(WK_Sim[1:-1])-1)

    # print(len(KM))



# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# Gamma = 0.1
# ThSim = 0.02
# nu = 2
# eta = 0.2

# Gamma = 0.1
# Prmk = 0
# nkern = 2
# gamma = 0.2
# eta = 0.2
# nu = 2
# kard = 3
# ThSim = 0.02
# Process = 0
# WK_Sim= [[0,0.2],[1,0.02]]
# from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
# # from Kernel_Decision_Function import Kernel_Decision_Function
# KM0 = []

# KM1 = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1), Gamma, nu, eta,KM0)

# KM =np.append(KM1,Kernel_Machine_Initialisation(X[1,:].reshape(1,-1), Gamma, nu, eta,KM1))
# # print(len(KM))
# from Kernel_Decision_Function import Kernel_Decision_Function
# WK_Sim = Kernel_Decision_Function(X[1,:].reshape(1,-1) ,ThSim, Gamma,KM)

# Robust_Kernel_Machine(Process,WK_Sim,X[1,:].reshape(1,-1),gamma,nu,eta,kard,KM)
