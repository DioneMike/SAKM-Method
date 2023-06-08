import numpy as np
import pandas as pd
from Kernel_Similarity_Mesure import Kernel_Similarity_Mesure
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation

def Kernel_Decision_Function(Xnew, ThSim, gamma,KM):

    global Rcum # Risque cumulé
    # global KM
    #global KM # une liste de noyaux  (ici vecteur de noyaux)
    
    Rcum = []
    #KM = 
    
    WinK_SimK = np.empty((0, 2))  #liste de noyaux gagnant (leurs indices dans le vecteur de noyaux KM et leur valeur de similarité avec les données Xnew)
    a = 0 #
    
    if len(KM)==0:
        WinK_SimK = np.empty((0, 2))
           
    else:
        m = 0
        while m < len(KM) :
            
             
            [dskm, fval ] = Kernel_Similarity_Mesure(Xnew, KM[m], gamma)
            
            # print("tiak tiak tiakkk",[dskm, fval ])
            if (np.abs(dskm) <= ThSim) and (fval is not None):
                
                new_row = np.array([int(m), fval])
               
                WinK_SimK = np.append(WinK_SimK, [new_row], axis=0) #probléme quand je mets plusieurs noyaux
                
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


# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# Gamma = 0.1
# ThSim = 0.02
# nu = 0.5
# eta = 0.2
# KM0 = []
# KM1 = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1) , Gamma, nu, eta,KM0)
# KM = Kernel_Machine_Initialisation(X[5,:].reshape(1,-1) , Gamma, nu, eta,KM1)
# # print(len(KM))
# K = Kernel_Decision_Function(X[3,:].reshape(1,-1) ,ThSim, Gamma,KM1)

# print(K)