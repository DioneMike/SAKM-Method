import numpy as np
import pandas as pd
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
import math


def KernelUpdate_NormTruncSGradient(WinKern,Xnew,Gamma,nu,eta,Kard,KM):

   
    
    if len(KM) == 0 :
        print("Il n'y a aucun noyau à mettre à jour !")
        return
        
    else :
        
        WKern = KM[int(WinKern[0][0])] # le noyau gagnant
        
        
        
        Smval = WinKern[0][1] # la valeur de similarité du noyau gagnant (ou le cluster gagnant)
        
        #--------------------------  Processus de mis à jour du noyau gagnant ----------------------------------------------#
    
        if Smval < 0: # cas de vecteur de support éléments d'erreur de marge (=> delta = 1 => muPhi non nul(voir page 5) => il y a similarité)
            
            
            
            WKern["Xsv"] = np.append(WKern["Xsv"],Xnew, axis=0) # mis à jour des vecteurs support
            
            WKern["wgh"] = np.append((1-eta)*WKern["wgh"], eta) # mis à jour des poids ; ajouter eta à la liste (1-eta)*Wkern.Wgh
            
            WKern["Rho"] = WKern["Rho"] + eta*(1-nu) # mis à jour du décalage 
            
            WKern["idsv"] = np.append(WKern["idsv"] , WKern["data"].shape[0]) # mis à jour des indices de SV
            
            WKern["data"] = np.append(WKern["data"], Xnew, axis=0) # mis à jour des données du noyau
        else :
            WKern["wgh"] = (1-eta)*WKern["wgh"]
            WKern["Rho"] =  WKern["Rho"] - eta*nu
            
            WKern["data"] = np.append(WKern["data"], Xnew,axis=0)
            
    
        #---------------------------------------- Troncature ------------------------------------------------------------------#
    
        if WKern["Xsv"].shape[0] == Kard :
            
           WKern["data"]= np.delete(WKern["data"], 0, axis=0)
           
           
        if WKern["Xsv"].shape[0] > Kard :
            iw = 0
            WKern["data"]= np.delete(WKern["data"], iw, axis=0)
            WKern["Xsv"] = np.delete(WKern["Xsv"], iw, axis=0)
            WKern["wgh"] =np.delete(WKern["wgh"], iw, axis=0)
            WKern["idsv"] =np.delete(WKern["idsv"], iw, axis=0)
            
     
          
        #----------------------------------------Condition suplémentaire: sum(alpha_i) = 1 et Rho = f(Xsvi)  --------------------#
    
        ts =  max(math.floor(WKern["wgh"].shape[0] / 2), 0)-1
       
        WKern["wgh"] = WKern["wgh"]/np.sum(WKern["wgh"]) # normalisation des poids alpha
        
       
        MXsv = np.ones((WKern["wgh"].shape[0], 1)).dot((WKern["Xsv"][ts,:]).reshape(1,-1))
        M = (WKern["Xsv"]-MXsv).dot((WKern["Xsv"]-MXsv).T)
        
        dist = np.diag(M)
        ksvi = np.exp(-Gamma*dist)
        WKern["Rho"] = (WKern["wgh"]).T.dot(ksvi)
        KernUpdat = WKern

    return KernUpdat


# data = pd.read_csv('Env_nonstat.csv', sep='\t')
# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)
# Gamma = 0.1

# X = pd.read_csv('datatest.csv',sep=";", header=None)
# X = np.array(X)
# X1 = X[0,:].reshape(1,-1)
# X2 = X[102,:].reshape(1,-1)

# gamma = 2
# nu = 0.3
# eta = 0.2
# kard = 50
# ThSim = 50
# Nc = 10; T = 30
# # # global KM
# # # KM = []

# # # print(X[1,:].reshape(-1,1).shape)
# # KM = Kernel_Machine_Initialisation(X, Gamma, nu, eta,[])
# # K = KernelUpdate_NormTruncSGradient([[0,0.2]],X,Gamma,nu,eta,kard,KM)
# Xini = X[2,:].reshape(1,-1)
# KM = Kernel_Machine_Initialisation(X1, gamma, nu, eta,[])
# from Kernel_Decision_Function import Kernel_Decision_Function
# w_k = Kernel_Decision_Function(X2, ThSim, gamma,KM)

# print(w_k)
# # # #print(.shape)

# K = KernelUpdate_NormTruncSGradient(w_k,X2,gamma,nu,eta,kard,KM)

# print(K)
# # # # # print(K["Xsv"].shape)
# # # # # print(K["data"].shape)


# # # # K = []
# # # # for i in range(30) :

# # # # # # KERN = Kernel_Machine_Initialisation(X, Gamma, nu, eta)
    
# # # #     K.append(KernelUpdate_NormTruncSGradient([[0,-0.2]],X[i,:].reshape(1,-1),Gamma,nu,eta,kard,KM))
    
# # # # print(np.sum(K[0]["wgh"]))