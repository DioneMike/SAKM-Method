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

    # global KM
    #global X
    
    if Process == 1 :
        return
    
    else:
    
        nkern = len(WK_Sim)
        # idkern = []
        
        
        # for v in WK_Sim :
            
        #     id = v[0]
        #     idkern.append(id)
        
        win = np.array(WK_Sim)
        
        Wnmk = np.sort(win[:,0]) #np.sort(idkern)
        print(Wnmk)
        
        # KERN = []
        
      
        
        KernMerg = Kernel_Merging_Function(KM[Wnmk],nkern, Wnmk[0],gamma,nu,eta,kard,KM)
     
        KernMergUpdat = KernelUpdate_NormTruncSGradient(WK_Sim,Xnew,gamma,nu,eta,kard,KM)
    
        #------------------------ Mis à jour du noyau----------------------------------#
        n = len(KM)-nkern + 1
        
        KM = np.resize(KM, n) 
        
        KM[int(Wnmk[0])] = KernMergUpdat
        
        
        for i in Wnmk[1:] :
            i = int(i)
            if len(KM) > i:
                KM =np.delete(KM,i)
            
            else :
                KM[int(Wnmk[0])] = KernMergUpdat


# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X)
# X0 = X[0,:].reshape(1,-1)
# X2 = X[1,:].reshape(1,-1)
# # Gamma = 0.1
# # nu = 0.5
# eta = 0.2

# # Gamma = 2
# # ThSim = 0.02
# # nu = 0.5
# # eta = 0.2

# Prmk = 0
# nkern = 2
# gamma = 2

# nu = 0.5
# kard = 50
# ThSim = 0.02
# Process = 0

# from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
# # from Kernel_Decision_Function import Kernel_Decision_Function
# KM0 = []

# KM1 = Kernel_Machine_Initialisation(X[0,:].reshape(1,-1), gamma, nu, eta,KM0)

# KM = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1), gamma, nu, eta,KM1)
# KM = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1), gamma, nu, eta,KM1)
# from Kernel_Decision_Function import Kernel_Decision_Function
# WK_Sim = Kernel_Decision_Function(X[1,:].reshape(1,-1) ,ThSim, gamma,KM)
# print(WK_Sim)
# Robust_Kernel_Machine(Process,WK_Sim,X[1,:].reshape(1,-1),gamma,nu,eta,kard,KM)
