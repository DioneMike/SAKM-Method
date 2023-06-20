
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Kernel_Decision_Function import Kernel_Decision_Function
from Online_Learning_Kernel import Online_Learning_Kernel
from Kernel_Machine_Initialisation import Kernel_Machine_Initialisation
from Eliminate_Noise_Clusters import Eliminate_Noise_Clusters
# from Plotting_Kernel_Clusters import Plotting_Kernel_Clusters
# from Plotting_Kernel_Function import Plotting_Kernel_Function
from Robust_Kernel_Machine import Robust_Kernel_Machine

# Charger le fichier .mat
# data = pd.read_csv('Env_nonstat.csv', sep='\t')

# xo = np.array(data)[1:2595,0]
# X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)


import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# # plt.plot(,,X[:,2])
data = pd.read_csv('datatst.csv',sep=",", header=None)
X = np.array(data)
x = X[:,0]
y = X[:,1]
z = X[:,2]

df = pd.DataFrame(data)

data=np.array(X)
# data= data[0:2595,0:2]#sio.loadmat(r'D:\Travaux\PhD_PROJECT\IMPLEMENTATIONS_CODES\DEVELOPPEMENTS\ADEME\ADEME Experiments\ADEME_Static_FN-EC-RE-FI_14-12-04.mat')
#X = data
#print(type(X))

#X = X[:-131, :]  # Supprimer les 130 dernières lignes de X
#print(X[:,0].shape)

# X = np.delete(X, 2,axis = 1)

# Afficher un graphique 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y,x,z, c='k', marker='x')
# ax.scatter(X[:, 0], X[:, 1], c='k', marker='x')
plt.show()

# Mesure de Performances en dynamique
global Rcum
Rcum = [0]

# global KM

# print(X.shape)

Xcl = np.empty((0,X.shape[1]))
gamma = 2
nu = 0.3
eta = 0.2
kard = 50
ThSim = 1
Nc = 10
T = 30

# Paramètres pour le graphique
echAxes = [-2, 10, -8, 12]
IT = 1
Step = 10
k = 1
Ndat = X.shape[0]
Axs = [0, 1]
Data = X.copy()
i = T

# Initialisation de Kernel Machine
KM = Kernel_Machine_Initialisation(X[0,:].reshape(1,-1), gamma, nu, eta,[])



while Data.shape[0] > 0:
    # Acquisition de données en ligne
    Xnew = Data[0, :].reshape(1, -1)
    
    # Data = Data[1:, :].reshape(1, -1)
    
    Data = np.delete(Data, 0, axis=0)
    
    # Fonction de décision pour Kernel Machine
    WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma,KM)  # À implémenter
    
   
    # Apprentissage en ligne avec Kernel Machine : Initialisation du noyau et adaptation
    Process = Online_Learning_Kernel(Xnew, WK_Sim, gamma, nu, eta, kard,KM)  # À implémenter

   
    
    # Machine à noyau robuste : Fusion du noyau
    
    Robust_Kernel_Machine(Process, WK_Sim, Xnew, gamma, nu, eta, kard,KM)  # À implémenter

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

# Tracé de Rcum
plt.figure()
plt.plot(Rcum, 'r')
plt.show()


R =[(1/(len(KM) * Ndat))* r for r in Rcum]

# Tracé de la fonction objective
plt.figure()
plt.title('Fonction objective : Erreur moyenne dynamique')
plt.axis([0, 2000, 0, 1])
plt.plot(R, 'r')
plt.show()


# import matplotlib.pyplot as plt

# # X = pd.read_csv('datatest.csv',sep=";", header=None)
# # X = np.array(X)
# xk = KM[0]["data"][:,0]
# yk = KM[0]["data"][:,1]
# zk = KM[0]["data"][:,2]

# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')
# ax.grid()

# ax.scatter(yk,xk,zk,c = 'r', s = 50)
# ax.set_title('2D Scatter Plot')

# # Set axes label
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# # ax.set_zlabel('z', labelpad=20)

# plt.show()

