import numpy as np
import pandas as pd
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation 
from KernelUpdate_NormTruncSGradient import KernelUpdate_NormTruncSGradient


def Online_Learning_Kernel(Xnew, WK_Sim,gamma,nu,eta,kard,KM):

    # >>INPUT
#     Xnew : nouvelle Observation acquise à l'instant 
#     WK_Sim : Noyau gagnant et Valeur de Similarity correspondante avec l'observation XT
#     gamma: Parametre du RBF, correspond à 1/2v² où v² est la variance des données
#     nu: Paramètre du Slack-error (admise), Règlage du nombre de SV et non-SV
#     eta: ratio d'apprentissage du processus d'Stochastic_Gradient
#     kernel (default) = 'gaussian' : Choix fixé du noyau gaussian RbF
#     kard :  Cardinalité max des noyaux
# >>OUTPUT
#     KM Kernel machine -> Structure modifié (variable global)
#     Process: etat 1 si la donnée est traité 0 si Ambiguité (Mergin)

    # global KM


    if len(WK_Sim)== 0 :

        # Initialisation du noyau : le premier noyau ou un nouveau noyau
        
        KM1 = Kernel_Machine_Initialisation(Xnew,gamma,nu,eta,KM)
        Process = 1  #Données traitée par le processus
        

    else :
        nw = len(WK_Sim)
        
        if nw == 1 :
            # Adaptation du noyau à l'aide de la technique de descente du gradient stochastique
    
            KernUpdat = KernelUpdate_NormTruncSGradient(WK_Sim,Xnew,gamma,nu,eta,kard,KM)
            
            id = int(WK_Sim[0][0])
            KM[id] = KernUpdat
            
            Process = 1  #Données traitée par le processus
            
        else :
            Process = 0  #Données pas encore traitée par le processus :ambiguité (Fusion)
        

    return Process

"""
WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma)

"""
# from Kernel_Decision_Function import Kernel_Decision_Function
# # data = pd.read_csv('Env_nonstat.csv', sep='\t')
# # xo = np.array(data)[1:2595,0]
# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X)
# Gamma = 2
# Prmk = 0
# nkern = 1
# gamma = 2
# eta = 0.2
# nu = 0.3
# kard = 50
# ThSim = 1
# X1 = X[0,:].reshape(1,-1)
# X2 = X[1,:].reshape(1,-1)
# KM0 = []
# KM = Kernel_Machine_Initialisation(X1,gamma,nu,eta,KM0)
# WK_Sim = Kernel_Decision_Function (X2, ThSim, gamma,KM)#[[0,0.2]]


# print(KM)
# p = Online_Learning_Kernel(X2, WK_Sim,gamma,nu,eta,kard,KM)
# print(p)