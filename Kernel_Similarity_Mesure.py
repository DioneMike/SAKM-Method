import numpy as np
import pandas as pd
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def Kernel_Similarity_Mesure(Xt,kern,gamma,meth=2):
    

   

############## Les entrées ################################
    """ Xt : l'observation à l'instant t
     kern : la fonction noyau pour l'évaluation de similarité 'nom_noyau'
     gamma : paramétre du noyau de type RBF
     meth : méthode d'évaluation de la similarité à utiliser, on le fixe par défaut à 2.
    Si l'utilsateur ne le fournit pas on le consdére par défaut  à 2"""
    
############# Les sorties ###############################

# dsk :la valeur de simularité entre l'observation XT et le noyau kern
# fval : la valeur prise par la fonction noyau (? est elle un vecteur ou un scalaire)

    
    # On considére kern comme un dictionnaire

    Rho = kern["Rho"]
    
    Wgh = kern["wgh"] #les poids 

   
   
    Xsv = kern["Xsv"] # les vecteurs de support
   
   
    # traitement par défaut
 
    Xmat = (np.ones((Wgh.shape[0], 1))).dot(Xt) 
   
    dist = np.diag((Xmat- Xsv).dot((Xmat - Xsv).T))
    
    Ksvt = np.exp(-gamma * dist)

    fval = np.dot(Wgh.T, Ksvt)-Rho # (valeur de la fonction noyau)
    
    
   
    if meth == 1:
        dsk = fval
    elif meth == 2:
        Xmat = (np.ones((Wgh.shape[0], 1))).dot(Xt) 
        
        dist = np.diag((Xmat- Xsv).dot((Xmat - Xsv).T))
        
        Ksvt = np.exp(-gamma * dist)
        
        if fval >= 0:
            dsk = 0
            
        else:
            dsk = np.min(np.sqrt(1 - Ksvt))
            
            
    else:
        KGrm = []
        for i in range(Wgh.shape[0]+1):
            
            Xmat = (np.ones((Wgh.shape[0], 1))).dot(Xt)
            
            dist = np.diag((Xmat- Xsv).dot((Xmat - Xsv).T))
            KGrm = np.column_stack((KGrm, dist))
        
        
        
        if fval >= 0:
            dsk = 0
            
        else:
            dsk = np.dot(np.dot(Wgh.T, KGrm), Wgh) - np.dot(Wgh.T, Ksvt) + 1
            
    dsk_fval = [dsk,fval] #retourne la valeur fonction d'apprentissage fval et la valeur de similarité dsk

    return dsk_fval

# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# X1 = X[1,:].reshape(1,-1)
# X2 = X[3:20,:].reshape(1,-1)
# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X)
# X0 = X[0,:].reshape(1,-1)
# X2 = X[1,:].reshape(1,-1)
# Gamma = 0.1
# nu = 0.5
# eta = 2

# KM0 = []
# kern = Kernel_Machine_Initialisation(X0 , Gamma, nu, eta,KM0)[0]
# print(kern)

# meth=2
# gamma=0.2
# v = Kernel_Similarity_Mesure(X2,kern,gamma,meth)
# print(v)
# for i in range(5) :
#     v = Kernel_Similarity_Mesure(X[i,:].reshape(1,-1),kern,gamma,meth)
#     print(v)
