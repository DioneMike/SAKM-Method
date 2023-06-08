
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
data = pd.read_csv('Env_nonstat.csv', sep='\t')

xo = np.array(data)[1:2595,0]
X =np.insert(np.array(data)[1:2595,0:2],1,xo,axis=1)





data=np.array(data)
# data= data[0:2595,0:2]#sio.loadmat(r'D:\Travaux\PhD_PROJECT\IMPLEMENTATIONS_CODES\DEVELOPPEMENTS\ADEME\ADEME Experiments\ADEME_Static_FN-EC-RE-FI_14-12-04.mat')
#X = data
#print(type(X))

#X = X[:-131, :]  # Supprimer les 130 dernières lignes de X
#print(X[:,0].shape)

# Afficher un graphique 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1],X[:, 2], c='k', marker='x')
plt.show()

# Mesure de Performances en dynamique
Rcum = [0]


# print(X.shape)

Xcl = np.empty((0,X.shape[1]))
gamma = 2
nu = 0.3
eta = 0.2
kard = 50
ThSim = 1
Nc = 2
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
X0 = X[1, :].reshape(1, -1)

km = Kernel_Machine_Initialisation(X0, gamma, nu, eta,[])

while Data.size > 0:
    # Acquisition de données en ligne
    Xnew = Data[0, :].reshape(1, -1)
    
    # Data = Data[1:, :].reshape(1, -1)
    
    Data = np.delete(Data, 0, axis=0)
   
    # Fonction de décision pour Kernel Machine
    WK_Sim = Kernel_Decision_Function(Xnew, ThSim, gamma,km)  # À implémenter
    
    
    # Apprentissage en ligne avec Kernel Machine : Initialisation du noyau et adaptation
    Process = Online_Learning_Kernel(Xnew, WK_Sim, gamma, nu, eta, kard,km)  # À implémenter

   
    
    # Machine à noyau robuste : Fusion du noyau
    
    Robust_Kernel_Machine(Process, WK_Sim, Xnew, gamma, nu, eta, kard,km)  # À implémenter

    # Élimination des clusters de noyau : Élimination du bruit
    if (k == i or k == Ndat) and len(km) > 0:
        Xcl, Xrj = Eliminate_Noise_Clusters(Nc,km)  # À implémenter

        i += T
    else:
        Xcl = np.append(Xcl, Xnew, axis=0)

    # Affichage des clusters de noyau
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


R =[(1/(len(km) * Ndat))* r for r in Rcum]

# Tracé de la fonction objective
plt.figure()
plt.title('Fonction objective : Erreur moyenne dynamique')
plt.axis([0, 2000, 0, 1])
plt.plot(R, 'r')
plt.show()
