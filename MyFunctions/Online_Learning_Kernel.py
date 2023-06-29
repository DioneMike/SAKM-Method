import numpy as np
import pandas as pd
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation 
from KernelUpdate_NormTruncSGradient import KernelUpdate_NormTruncSGradient


def Online_Learning_Kernel(Xnew, WK_Sim,gamma,nu,eta,kard):

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

    global KM
    

    WK_Sim = np.array(WK_Sim)
    if len(WK_Sim)== 0 :

        # Initialisation du noyau : le premier noyau ou un nouveau noyau
        
        KM = Kernel_Machine_Initialisation(Xnew,gamma,nu,eta,KM)
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
from Kernel_Decision_Function import Kernel_Decision_Function
from Robust_Kernel_Machine import Robust_Kernel_Machine
from Eliminate_Noise_Clusters import Eliminate_Noise_Clusters
# data = pd.read_csv('Env_nonstat.csv', sep='\t')
# xo = np.array(data)[1:2595,0]
X = pd.read_csv('datatst.csv',sep=",", header=None)
X = np.array(X)

Xcl = np.empty((0,X.shape[1]))
gamma = 2
nu = 0.3
eta = 0.2
kard = 50
ThSim = 0.55
Nc = 10
T = 30

# Paramètres pour le graphique
echAxes = [-2, 10, -8, 12]
IT = 1
Step = 10
k = 1
Ndat = X.shape[0]
Axs = [0, 1]
Data = X.copy()#[495:502,:]#0.reshape(1,-1)
i = T

# Initialisation de Kernel Machine
global KM
KM = Kernel_Machine_Initialisation(X[0,:].reshape(1,-1), gamma, nu, eta,[])



while Data.shape[0] > 0:
    # Acquisition de données en ligne
    Xnew = Data[0, :].reshape(1, -1)
    
    # Data = Data[1:, :].reshape(1, -1)
    
    Data = np.delete(Data, 0, axis=0)
    
    # Fonction de décision pour Kernel Machine
    WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma,KM)  # À implémenter
    
    
    # Apprentissage en ligne avec Kernel Machine : Initialisation du noyau et adaptation
    Process = Online_Learning_Kernel(Xnew, WK_Sim, gamma, nu, eta, kard)  # À implémenter

    # print(WK_Sim)
    # print(Process)
    
    # Machine à noyau robuste : Fusion du noyau
    
    # Robust_Kernel_Machine(Process, WK_Sim, Xnew, gamma, nu, eta, kard,KM)  # À implémenter

    # Élimination des clusters de noyau : Élimination du bruit
    if (k == i or k == Ndat) and len(KM) > 0:
       Xcl, Xrj = Eliminate_Noise_Clusters(Nc,KM)  # élimination de bruits
        
       i += T
    else:
        
        
         Xcl = np.append(Xcl, Xnew, axis=0) 
        
        


    # Affichage des clusters de noyau (J'ai des soucis de code pour cette partie)
    # if k == IT or k == Ndat:
    #     Plotting_Kernel_Clusters(Xcl, KM, gamma, echAxes, Axs)  # À implémenter
    #     plt.pause(0.2)
    #     Plotting_Kernel_Function(Xcl, KM, gamma, echAxes, Axs)  # À implémenter
    #     IT = k + Step
    #     plt.pause(0.2)
    
    k += 1