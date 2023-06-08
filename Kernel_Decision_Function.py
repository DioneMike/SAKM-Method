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
            
           
            if (np.abs(dskm) <= ThSim) and (fval !=0) :
                
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