# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:51:52 2023

@author: michel.dione
"""
import numpy as np
import pandas as pd
import math 
import plotly.io as pio
import plotly.graph_objs as go
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation 

def Kernel_Machine_Initialisation(Xini, Gamma, nu, eta):

    # Xini: représente les nouvelles observations acquises à l'instant 
    # Gamma : paramétre du noyau RBF
    # nu  : paramétres permettant de regler le nombre de vecteurs de support
    # eta : représente le tau d'apprentissage du processus du gradiant stochastique
    
     
    global KM
    
    
    
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


#Kernel similarity mesure function

def Kernel_Similarity_Mesure(Xt,kern,gamma,meth=2):
    

   

############## Les entrées ################################
    """ Xt : l'observation à l'instant t
     kern : la fonction noyau pour l'évaluation de similarité 'nom_noyau'
     gamma : paramétre du noyau de type RBF
     meth : méthode d'évaluation de la similarité à utiliser, on le fixe par défaut à 2.
    Si l'utilsateur ne le fournit pas on le consdére par défaut  à 2"""
    
############# Les sorties ###############################

# dsk :la valeur de simularité entre l'observation XT et le noyau kern
# fval : la valeur prise par la fonction noyau 

    # global KM
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
            dsk = 0 # la donnée est dans la même classe que le Xsv
            
        else:
            dsk = np.min(np.sqrt(1 - Ksvt)) # la distance entre la donnée et le Xsv qui est plus proche
            
            
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


# kernel update kernel machine 

def KernelUpdate_NormTruncSGradient(WinKern,Xnew,Gamma,nu,eta,Kard):

    global KM
    
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
    
        if WKern["Xsv"].shape[0] >= Kard :
            
           WKern["data"]= np.delete(WKern["data"], 0, axis=0)
           
           
        if WKern["Xsv"].shape[0] > Kard :
            iw = 0
            WKern["data"]= np.delete(WKern["data"], iw, axis=0)
            WKern["Xsv"] = np.delete(WKern["Xsv"], iw, axis=0)
            WKern["wgh"] =np.delete(WKern["wgh"], iw, axis=0)
            WKern["idsv"] =np.delete(WKern["idsv"], iw, axis=0)
            
     
          
        #----------------------------------------Condition suplémentaire: sum(alpha_i) = 1 et Rho = f(Xsvi)  --------------------#
    
        ts =  max(math.floor(WKern["wgh"].shape[0] / 2), 0)
       
        WKern["wgh"] = WKern["wgh"]/np.sum(WKern["wgh"]) # normalisation des poids alpha
        
       
        MXsv = np.ones((WKern["wgh"].shape[0], 1)).dot((WKern["Xsv"][ts,:]).reshape(1,-1))
        M = (WKern["Xsv"]-MXsv).dot((WKern["Xsv"]-MXsv).T)
        
        dist = np.diag(M)
        ksvi = np.exp(-Gamma*dist)
        WKern["Rho"] = (WKern["wgh"]).T.dot(ksvi)
        KernUpdat = WKern

    return KernUpdat

# kernel Merging function 

def Kernel_Merging_Function(KERN, nkern,Prmk, gamma, nu, eta, kard) :

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


    
    global KM
    
     # pour l'ensemble des noyaux
    X = np.empty((0,KERN[0]["data"].shape[1])) # pour stocker  l'ensemble des données fusionnées 
    
    
    for m in range(1,len(KERN)):
       
        X = np.append(X, KERN[m]['data'],axis=0)
        
    KernUpdat = KM[Prmk] # on initialise avec le premier noyau
     
    for i in range(X.shape[0]) :
        Xnew = X[i,:].reshape(1,-1)
        
        
        [dsk,fval] = Kernel_Similarity_Mesure(Xnew,KernUpdat,gamma,2)
        
        KernUpdat = KernelUpdate_NormTruncSGradient([[Prmk,fval]],Xnew,gamma,nu,eta,kard)
        
        
        KM[Prmk] = KernUpdat
        

    KernMerg = KernUpdat
    # print("___________________",KernMerg)
    
    return KernMerg


# Kernel decision function

def Kernel_Decision_Function(Xnew, ThSim, gamma):

    global Rcum # Risque cumulé
    global KM
    
    
    
    
   
    
    
    
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
               
                WinK_SimK = np.append(WinK_SimK, [new_row], axis=0) 
                
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


#kernel online learning function 
def Online_Learning_Kernel(Xnew, WK_Sim,gamma,nu,eta,kard):

    """ 
    >>Entrées
    Xnew : nouvelle Observation acquise à l'instant 
    WK_Sim : Noyau gagnant et Valeur de Similarity correspondante avec l'observation XT
    gamma: Parametre du RBF, correspond à 1/2v² où v² est la variance des données
    nu: Paramètre du Slack-error (admise), Règlage du nombre de SV et non-SV
    eta: ratio d'apprentissage du processus d'Stochastic_Gradient
    kernel (default) = 'gaussian' : Choix fixé du noyau gaussian RbF
    kard :  Cardinalité max des noyaux
    >>Sorties
    KM Kernel machine -> Structure modifié (variable global)
    Process: etat 1 si la donnée est traité 0 si Ambiguité (Mergin)
    """

    global KM
    WK_Sim = np.array(WK_Sim)
    if len(WK_Sim)== 0 :

        # Initialisation du noyau : le premier noyau ou un nouveau noyau
        
        KM = Kernel_Machine_Initialisation(Xnew,gamma,nu,eta)
        Process = 1  #Données traitée par le processus
        

    else :
        nw = len(WK_Sim)
        
        if nw == 1 :
            # Adaptation du noyau à l'aide de la technique de descente du gradient stochastique
    
            KernUpdat = KernelUpdate_NormTruncSGradient(WK_Sim,Xnew,gamma,nu,eta,kard)
            
            id = int(WK_Sim[0][0])
            KM[id] = KernUpdat
            
            Process = 1  #Données traitée par le processus
            
        else :
            Process = 0  #Données pas encore traitée par le processus :ambiguité (Fusion)
        
    
    return Process


#Kernel robust function 


def Robust_Kernel_Machine(Process,WK_Sim,Xnew,gamma,nu,eta,kard):
# >>INPUT:
#     Process: etat 0 si la donnée est traité 1 si Ambiguité (Mergin)
#     WK_Sim : Noyau gagnant et Valeur de Similarity correspondante avec l'observation XT
#     Xnew : nouvelle Observation acquise à l'instant 
#     gamma: Parametre du RBF, correspond à 1/2v² où v² est la variance des données
#     nu: Paramètre du Slack-error (admise), Règlage du nombre de SV et non-SV
# >>OUTPUT:

    global KM
    #global X
    
    if Process == 1 :
        return
    
    else:
    
        nkern = len(WK_Sim)
        # idkern = []
        
        
        # for v in WK_Sim :
            
        #     id = v[0]
        #     idkern.append(id)
        
        win = np.array(WK_Sim)
        
        Wnmk = np.sort(win[:,0]) #np.sort(idkern)
        # print(Wnmk)
        
        KERN = []
        for i in Wnmk :
            KERN.append(KM[int(i)])
        
      
        
        KernMerg = Kernel_Merging_Function(KERN,nkern, int(Wnmk[0]),gamma,nu,eta,kard)
     
        KernMergUpdat = KernelUpdate_NormTruncSGradient(WK_Sim,Xnew,gamma,nu,eta,kard)
    
        #------------------------ Mis à jour du noyau----------------------------------#
        n = len(KM)-nkern + 1
        
        KM = np.resize(KM, n) 
        
        KM[int(Wnmk[0])] = KernMergUpdat
        
        
        for i in Wnmk[1:] :
            i = int(i)
            if len(KM) > i:
                KM =np.delete(KM,i)
            
            else :
                KM[int(Wnmk[0])] = KernMergUpdat


#kernel elimination function

def Eliminate_Noise_Clusters(Nc):
    global KM
    Xrj = np.empty((0,KM[0]["data"].shape[1]))
    Xcl = np.empty((0,KM[0]["data"].shape[1]))
    m = 0

    while m < len(KM):
        
        if KM[m]["data"].shape[0] < Nc :
            Xrj = np.append(Xrj,KM[m]["data"])
            # KM[m] = []
            KM = np.delete(KM,m)
            
        else :
            Xcl = np.append(Xcl,KM[m]["data"],axis=0)

        m = m + 1
    # print(KM)
    return Xcl,Xrj


# main  SAKM

import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# # plt.plot(,,X[:,2])

data = pd.read_csv('datatst.csv',sep=",", header=None) #les données de test

# data = pd.read_csv('Env_nonstat.csv',sep=",", header=None) # les données non stationnaires
X = np.array(data) # tableau à deux dimensions
# xo = X[0:2596,0]
# X =np.insert(X,2,xo,axis=1)[:,0:3] # on rajoute une troisiéme dimension.



data=np.array(X)


# Mesure de Performances en dynamique
global Rcum
Rcum = [0]

global KM
KM = [] #l'espace des classe est initialement vide 



Xcl = np.empty((0,X.shape[1]))
gamma = 2
nu = 0.3
eta = 0.2
kard =120
ThSim = 0.99
Nc = 10
T = 30

# Paramètres pour le graphique
echAxes = [-2, 10, -8, 12]
IT = 1
Step = 10
k = 1
Ndat = X.shape[0]
Axs = [0, 1]
# Data = X.copy()
Data = Data = X#[498:502,:]
i = T

# Représentation graphique des observations

x = X[:,0]
y = X[:,1]
z = X[:,2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(y, x,z, c='k', marker='x')
# # ax.scatter(X[:, 0], X[:, 1], c='k', marker='x')
# plt.show()


def plot_data():
    # pio.renderers.default = 'svg' # To plot in default 
    pio.renderers.default = 'browser' #To plot in browser  
    #Lets use plotly to plot the data. Plotly allows us to make an interactive chart. 
    fig = go.Figure() #Create empty figure
    #add_trace method to add plots to the figure
    #create a line plot with 'Close' as legend, date column as x-axis and Close column as y-axis. 
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               mode='markers', name='data', marker=dict(size=2)))
    fig.update_layout(showlegend = True)
    return fig.show()

plot_data() #Show plot



# Apprentissage séquentiel 

while Data.size > 0 :
    # Acquisition de données en ligne
    Xnew = Data[0, :].reshape(1, -1)
    # print(k,Xnew)
    # Data = Data[1:, :].reshape(1, -1)
    
    Data = np.delete(Data, 0, axis=0)
    
    # Fonction de décision pour Kernel Machine
    WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma)  # À implémenter
    
    # Apprentissage en ligne avec Kernel Machine : Initialisation du noyau et adaptation
    Process = Online_Learning_Kernel(Xnew, WK_Sim, gamma, nu, eta, kard)  # À implémenter

    # Machine à noyau robuste : Fusion du noyau
    Robust_Kernel_Machine(Process, WK_Sim, Xnew, gamma, nu, eta, kard)  # À implémenter

    # Élimination des clusters de noyau : Élimination du bruit
    if (k == i or k == Ndat) and len(KM) > 0:
        Xcl, Xrj = Eliminate_Noise_Clusters(Nc)  # élimination de bruits
        
        i += T
    else:
        
        
        Xcl = np.append(Xcl, Xnew, axis=0) 
        
        


   # partie réservée pour les fonctions d'affichages des clusters
    
    k += 1

# Tracé de Rcum
# plt.figure()
# plt.plot(Rcum, 'r')
# plt.show()


# R =[(-1/(len(KM) * Ndat))* r for r in Rcum]

# # Tracé de la fonction objective
# plt.figure()
# plt.title('Fonction objective : Erreur moyenne dynamique')
# # plt.axis([0, 2000, 0, 1])
# plt.plot(R, 'r')
# plt.show()


#----------------------Partie vérification graphique--------------------#

# On vérifie graphiquement si nos deux classes données par l'algorithme correspondent aux deux classes observées
# data
x1 = KM[0]["data"][:,0]
y1 = KM[0]["data"][:,1]
z1 = KM[0]["data"][:,2]

# Xsv


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(y1, x1,z1, c='k', marker='x')
# # ax.scatter(X[:, 0], X[:, 1], c='k', marker='x')
# plt.show()


x2 = KM[1]["data"][:,0]
y2 = KM[1]["data"][:,1]
z2 = KM[1]["data"][:,2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(y2, x2,z2, c='k', marker='x')
# # ax.scatter(X[:, 0], X[:, 1], c='k', marker='x')
# plt.show()

def plot_cluster():
    # pio.renderers.default = 'svg' #To plot in default 
    pio.renderers.default = 'browser' #To plot in browser  
    #Lets use plotly to plot the data. Plotly allows us to make an interactive chart. 
    fig = go.Figure() #Create empty figure
    #add_trace method to add plots to the figure
    #create a line plot with 'Close' as legend, date column as x-axis and Close column as y-axis. 
    fig.add_trace(go.Scatter3d(x = x1, y = y1, z = z1, 
                               mode = 'markers', name = 'C1', marker=dict(size=2)))
    fig.add_trace(go.Scatter3d(x = x2, y = y2, z = z2, 
                               mode = 'markers', name = 'C2', marker=dict(size=2)))
    fig.update_layout(showlegend = True)
    
    
    return fig.show()

plot_cluster() #Show plot