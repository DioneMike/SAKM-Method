import numpy as np
import pandas as pd
from Kernel_Similarity_Mesure import Kernel_Similarity_Mesure
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation

def Kernel_Decision_Function(Xnew, ThSim, gamma,KM):

    global Rcum # Risque cumulé
   
    Rcum = [0]
    
    
    
   
    
    
    
    WinK_SimK = np.empty((0, 2))  #liste de noyaux gagnant (leurs indices dans le vecteur de noyaux KM et leur valeur de similarité avec les données Xnew)
    a = 0 #
    
    if len(KM)==0:
        WinK_SimK = np.empty((0, 2))
           
    else:
        m = 0
        while m < len(KM) :
            
             
            [dskm, fval] = Kernel_Similarity_Mesure(Xnew, KM[m], gamma)
            
           
            if (np.abs(dskm) <= ThSim) and (fval is not None) :
                
                new_row = np.array([int(m), fval])
               
                WinK_SimK = np.append(WinK_SimK, [new_row], axis=0) #probléme quand je mets plusieurs noyaux
                print("dshbnvc vcnv ")
            else :
                
                WinK_SimK = WinK_SimK
                
            m = m + 1
            if fval < 0 :
                a = a + fval
            
        
    if len(Rcum) == 0:
        
        Rcum.append(a)
    else:
        Rcum.append(Rcum[-1]+ a)  
    
    return WinK_SimK

    # Cas 1 : WinK_SimK = []; aucun noyau similaire
    # Cas 2 : WinK_SimK = [[m, dskm]]; un seul noyau gagnant
    # Cas 3 : WinK_SimK = [[m1, dskm1], [m2, dskm2], ...] plusieurs noyaux gagnants


# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X)
# X0 = X[0,:].reshape(1,-1)
# Gamma = 2
# ThSim = 0.02
# nu = 0.5
# eta = 0.2
# # global KM
# global Rcum
# Rcum=[0]
# # KM = []
# KM0 = []
# KM1 = Kernel_Machine_Initialisation(X[0,:].reshape(1,-1) , Gamma, nu, eta,KM0)
# KM = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1) , Gamma, nu, eta,KM1)
# # print(len(KM))
# K = Kernel_Decision_Function(X[1,:].reshape(1,-1) ,ThSim, Gamma,KM)

# print(K)


# from Kernel_Decision_Function import Kernel_Decision_Function
# # data = pd.read_csv('Env_nonstat.csv', sep='\t')
# # xo = np.array(data)[1:2595,0]
# df = pd.read_csv('datatst.csv',sep=",", header=None)
# X = np.array(df)
# gamma = 2
# nu = 0.5
# eta = 0.2
# kard = 50
# ThSim = 1
# Nc = 10
# T = 30

# # Paramètres pour le graphique
# echAxes = [-2, 10, -8, 12]
# IT = 1
# Step = 10
# k = 0
# Ndat = X.shape[0]
# Axs = [0, 1]
# Data = X.copy()
# # Data = Data[0:10,:]
# i = T

# KM = Kernel_Machine_Initialisation(X[0,:].reshape(1,-1), gamma, nu, eta,[])

# while Data.size > 0 :

#     # Acquisition de données en ligne
#     Xnew = Data[0, :].reshape(1, -1)
    
#     # Data = Data[1:, :].reshape(1, -1)
    
#     Data = np.delete(Data, 0, axis=0)
    
#     # Fonction de décision pour Kernel Machine
#     WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma,KM)  # À implémenter
#     # print(Xnew)
#     # print(k,WK_Sim)
#     # Apprentissage en ligne avec Kernel Machine : Initialisation du noyau et adaptation
    

#     # print(WK_Sim)
    
#     # Machine à noyau robuste : Fusion du noyau
    
#     # Robust_Kernel_Machine(Process, WK_Sim, Xnew, gamma, nu, eta, kard,KM)  # À implémenter

#     # Élimination des clusters de noyau : Élimination du bruit
#     # if (k == i or k == Ndat) and len(KM) > 0:
#     #     Xcl, Xrj = Eliminate_Noise_Clusters(Nc,KM)  # élimination de bruits
        
#     #     i += T
#     # else:
        
        
#     #     Xcl = np.append(Xcl, Xnew, axis=0) 
        
        


#     # Affichage des clusters de noyau (J'ai des soucis de code pour cette partie)
#     # if k == IT or k == Ndat:
#     #     Plotting_Kernel_Clusters(Xcl, KM, gamma, echAxes, Axs)  # À implémenter
#     #     plt.pause(0.2)
#     #     Plotting_Kernel_Function(Xcl, KM, gamma, echAxes, Axs)  # À implémenter
#     #     IT = k + Step
#     #     plt.pause(0.2)
    
#     k += 1