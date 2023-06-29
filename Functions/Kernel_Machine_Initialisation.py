import numpy as np
import pandas as pd

# Dans ce programme nous allons créer une fonction qui permettra d'initialiser l'ensemle noyau noté  KM de la machine



def Kernel_Machine_Initialisation(Xini, Gamma, nu, eta,KM):

    # Xini: représente les nouvelles observations acquises à l'instant 
    # Gamma : paramétre du noyau RBF
    # nu  : paramétres permettant de regler le nombre de vecteurs de support
    # eta : représente le tau d'apprentissage du processus du gradiant stochastique
    
     
   
    
    
    
    Xsv = Xini
    
    wgh = np.ones(Xsv.shape[0])
    Rho = eta*(1-nu)
    idsv = 0
    data = Xini
    kern={"wgh":wgh, "Xsv":Xini, "Rho":Rho, "idsv":idsv, "data":data}
   
    if len(KM) == 0 : 
        # initialisation du premier noyau de la machine
        # un noyau (une classe) est caractérisé par ses paramétres (poids (Wgh), coef d'ajustement (Rho), supports vecteurs (Xsv), ses données (data))
        KM = np.append(KM,kern)
        

    else : 
       
        
        KM = np.append(KM, kern)
    
    

    return KM  
    


# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# Gamma = 0.1
# nu = 0.5
# eta = 2

# global KM
# KM= []

# KMInit = Kernel_Machine_Initialisation(X[1,:].reshape(1,-1) , Gamma, nu, eta,KM)

# print(KMInit)        




    