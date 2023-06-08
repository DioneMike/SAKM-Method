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

    #global KM


    if len(WK_Sim)== 0 :

        # Initialisation du noyau : le premier noyau ou un nouveau noyau
        KM0 = []
        KM = Kernel_Machine_Initialisation(Xnew,gamma,nu,eta,KM0)
        Process = 1  #Données traitée par le processus
        

    else :
        nw = len(WK_Sim)
        # print(WK_Sim)
        if nw == 1 :
            # Adaptation du noyau à l'aide de la technique de descente du gradient stochastique
    
            KernUpdat = KernelUpdate_NormTruncSGradient(WK_Sim,Xnew,gamma,nu,eta,kard,KM)
            
            id = int(WK_Sim[0][0])
            KM[id] = KernUpdat
            # print(KernUpdat)
            Process = 1  #Données traitée par le processus
    
        else :
            Process = 0  #Données pas encore traitée par le processus :ambiguité (Fusion)
        

    return Process

"""
WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma)

"""

# data = pd.read_csv('Env_nonstat.csv', sep='\t')
# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# Gamma = 0.1
# Prmk = 0
# nkern = 1
# gamma = 0.2
# eta = 0.03
# nu = 0.1
# kard = 3
# Xnew = X[1,:].reshape(-1,1)
# WK_Sim =[[0,0.2]]

# KM0 = []
# KM = Kernel_Machine_Initialisation(Xnew,gamma,nu,eta,KM0)
# p = Online_Learning_Kernel(Xnew, WK_Sim,gamma,nu,eta,kard,KM)
# print(p)