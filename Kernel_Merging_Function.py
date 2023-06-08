#Dans ce progarmme nous allons programmer une fonction qui permet la fusion de noyaux similaires

import numpy as np
import pandas as pd
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
from KernelUpdate_NormTruncSGradient import KernelUpdate_NormTruncSGradient
from Kernel_Similarity_Mesure import Kernel_Similarity_Mesure

def Kernel_Merging_Function(KERN, nkern,Prmk, gamma, nu, eta, kard,KM) :

    # Les entrées 

    '''
    KERN :  l'ensemble des noyaux à fusionner
    nkern : le nombre de noyaux à fusionner
    Prmk : le premier de la liste des noyaux à fusionner
    gamma :  paramétre du RBF
    nu : paramétre du slack_error, reglage du nombre de supports vecteur
    eta : le taux d'apprentissage dans la descente du gradient

    '''

    # La sortie 

    ''' Cette fonction va retourner KernMerg, le noyau résultant de la fusion des noyaux similaires '''


    
    
    
     # pour l'ensemble des noyaux
    X = np.empty((0,KERN[0]["data"].shape[1])) # pour stocker  l'ensemble des données fusionnées 
    #KM = Kernel_Machine_Initialisation(X, gamma, nu, eta)
    
    for m in range(len(KERN)):
        # print(KERN[0]["data"].shape, KERN[1]["data"].shape)
        #print(X.shape)
        X = np.vstack((X, KERN[m]['data']))
        #print(X.shape)
    KernUpdat = KM[Prmk] # on initialise avec le premier noyau
     
    for i in range(X.shape[0]) :
        Xnew = X[i,:].reshape(1,-1)
        
        # print(Xnew)
        [dsk,fval] = Kernel_Similarity_Mesure(Xnew,KernUpdat,gamma,2)
        #print([dsk,fval])
        KernUpdat = KernelUpdate_NormTruncSGradient([[Prmk,fval]],Xnew,gamma,nu,eta,kard,KM)
        
        print(KernUpdat)
        KM[Prmk] = KernUpdat
        

    KernMerg = KernUpdat
    # print("___________________",KernMerg)
    
    return KernMerg


# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)

# Xtest = X[1:10,:].reshape(1,-1)
# Gamma = 0.1
# Prmk = 0
# nkern = 1
# gamma = 0.2
# eta = 0.03
# nu = 0.1
# kard = 3

# KM0 = []
# KM1 = Kernel_Machine_Initialisation(Xtest, Gamma, nu, eta,KM0)
# KM2 = Kernel_Machine_Initialisation(Xtest, Gamma, nu, eta,KM1)
# KM3 = Kernel_Machine_Initialisation(Xtest, Gamma, nu, eta,KM2)
# KERN = KM2
# K = Kernel_Merging_Function(KERN, nkern,Prmk, gamma, nu, eta, kard,KM3)
# # print(K)