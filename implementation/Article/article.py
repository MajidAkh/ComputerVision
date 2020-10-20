from __future__ import print_function

from data_utils import load_CIFAR10
from reseau import *
import matplotlib.pyplot as plt
import time
import numpy as np
import warnings
import os.path
warnings.filterwarnings("ignore", "Mean of empty slice")
np.seterr(divide='ignore', invalid='ignore')

start_time = time.time()

def get_CIFAR10_data(num_training = 5000, num_validation = 1000, num_test = 500):
    """ On télécharge ici à partir du dossier et on prepare les données à être recu par le reseau de neuronne
  
    """

    # Chargerment des données brutes.
    cifar10_dir = '../../datasets/cifar-10-batches-py/'
    print(cifar10_dir)
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Sous ensemble des données
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalisation des données, on soustrait la moyenne.
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val = X_val - mean_image
    X_test = X_test - mean_image
    X_train = X_train.swapaxes(1,3)
    X_val = X_val.swapaxes(1,3)
    return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print("chargement terminé")
print("donnée d'entrainement :" , X_train.shape)
print("donnée de validation :" , X_val.shape)
print("donnée de test :" , X_test.shape)
print("Temps :" , (time.time()-start_time)/60)



rec_field_size = 6
nombre_centroide = 1600
Blanchement = True
nombre_patches = 400000
DIM_IMG = [32,32,3]



#création des patches
patches = []
for i in range(nombre_patches):
    if(np.mod(i, 10000)== 0):
        print("echantillonage pour kmeans",i,"/", nombre_patches)
    start_r = np.random.randint(DIM_IMG[0] - rec_field_size)
    start_c = np.random.randint(DIM_IMG[1] - rec_field_size)
    patch = np.array([])
    img = X_train[np.mod(i, X_train.shape[0])]
    for layer in img:
        patch = np.append(patch, layer[start_r:start_r + rec_field_size].T[start_c:start_c + rec_field_size].T.ravel())
    patches.append(patch)
patches = np.array(patches)

#on normalise les patches
patches = (patches-patches.mean(1)[:,None])/np.sqrt(patches.var(1)+ 10)[:, None]
print("time", (time.time()-start_time)/60)

del X_train, y_train, X_val, y_val, X_test, y_test

#blanchiment
print("Blanchiment")
[D,V]= np.linalg.eig(np.cov(patches, rowvar = 0))
P = V.dot(np.diag(np.sqrt(1/(D + 0.1)))).dot(V.T)
patches = patches.dot(P)

print("time", (time.time() - start_time)/60.0)
del D,V
#Application de K-means sur les patches
centroids = np.random.randn(nombre_centroide, patches.shape[1])*.1
num_iters = 50
batch_size = 128
for ite in range(num_iters):
    print("kmeans iters", ite+1,"/", num_iters )
    hf_c2_sum = .5*np.power(centroids, 2).sum(1)
    counts = np.zeros(nombre_centroide)
    summation = np.zeros_like(centroids)
    for i in range(0, len(patches), batch_size):
        last_i = min(i+batch_size, len(patches))
        idx = np.argmax(patches[i:last_i].dot(centroids.T) -hf_c2_sum.T, axis = 1)
        S = np.zeros([last_i - i, nombre_centroide])
        S[range(last_i-i), np.argmax(patches[i:last_i].dot(centroids.T)-hf_c2_sum.T, axis=1)]=1
        summation+=S.T.dot(patches[i:last_i])
        counts+= S.sum(0)
    centroids = summation/counts[:,None]
    centroids[counts==0]=0

print("time", (time.time()-start_time)/60.0)



def sliding(img, window=[6,6]):
    """ fonction qui permettrait le decoupage en patch des images.  """
    out = np.array([])
    for i in range(3):
        s = img.shape
        row = s[1]
        col = s[2]
        col_extent = col - window[1]+ 1
        row_extent = row - window[0]+ 1
        start_idx = np.arange(window[0])[:,None]*col + np.arange(window[1])
        offset_idx = np.arange(row_extent)[:,None]*col + np.arange(col_extent)
        if len(out)==0:
            out = np.take(img[i],start_idx.ravel()[:,None] + offset_idx.ravel())
        else:
            out=np.append(out,np.take(img[i], start_idx.ravel()[:,None] + offset_idx.ravel()),axis=0)
    return out


#extraction des features.
def extract_features(X_train):
    trainXC = []
    idx = 0
    for img in X_train:
        idx += 1
        if not np.mod(idx,1000):
            print('extract feature', idx, "/", len(X_train))
            print("time", (time.time()-start_time)/60)
        patches = sliding(img,[rec_field_size, rec_field_size]).T
        #on normalise
        patches = (patches-patches.mean(1)[:,None])/(np.sqrt(patches.var(1)+0.1)[:,None])
        patches = patches.dot(P)

        x2 = np.power(patches,2).sum(1)
        c2 = np.power(centroids,2).sum(1)
        xc = patches.dot(centroids.T)

        dist = np.sqrt(-2*xc+x2[:,None] + c2)
        u = dist.mean(1)
        #f_k(x) = max{0, mu(z) - z_k}
        patches = np.maximum(-dist+u[:, None],0)
        rs = DIM_IMG[0]-rec_field_size+1
        cs = DIM_IMG[1]-rec_field_size+1
        patches = np.reshape(patches, [rs, cs, -1])
        q = []
        q.append(patches[0:int(rs/2), 0:int(cs/2)].sum(0).sum(0))
        q.append(patches[0:int(rs/2), int(cs/2):int(cs-1)].sum(0).sum(0))
        q.append(patches[int(rs/2):int(rs-1),0:int(cs/2)].sum(0).sum(0))
        q.append(patches[int(rs/2):int(rs-1),int(cs/2):int(cs-1)].sum(0).sum(0))
        q = np.array(q).ravel()
        trainXC.append(q)

    trainXC = np.array(trainXC)
    
    trainXC=(trainXC-trainXC.mean(1)[:,None])/(np.sqrt(trainXC.var(1)+10)[:,None])
    return trainXC





X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
trainXC = extract_features(X_train)

print("time", (time.time()-start_time)/60.0)
valXC = extract_features(X_val)
testXC = extract_features(X_test)





input_size = trainXC.shape[1]
hidden_size = 500
num_classes = 10

net = TwoLayerNet(input_size, hidden_size, num_classes,1e-4)
stats = net.train(trainXC, y_train, valXC, y_val,num_iters=12000, batch_size=128,learning_rate=5e-4, learning_rate_decay=0.99,reg=0, verbose=True,update="momentum",arg=0.95,dropout=0.5)


val_acc = (net.predict(trainXC) == y_train).mean()
print ('Précision sur les donnéés dentrainement: ', val_acc)
val_acc = (net.predict(valXC) == y_val).mean()
print ('Précision réel: ', val_acc)

print ("time",(time.time()-start_time)/60.0)


