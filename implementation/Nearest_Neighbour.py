
from __future__ import print_function
from data_utils import load_CIFAR10
import random
import numpy as np
import matplotlib.pyplot as plt
.

plt.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'




Xtr, Ytr, Xte, Yte = load_CIFAR10('../datasets/cifar-10-batches-py/')

num_training = 5000
mask = list(range(num_training))
Xtr = Xtr[mask]
Ytr = Ytr[mask]

num_test = 500
mask = list(range(num_test))
Xte = Xte[mask]
Yte = Yte[mask]


50000 x 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
10000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

class NearestNeighbor(object):
    def __init__(self):

        pass

    def train(self, X, y):

        self.Xtr=X
        self.Ytr=y

    def predict(self, X):

        num_test=X.shape[0]

        Ypred=np.zeros(num_test, dtype=self.Ytr.dtype)


        for i in range(num_test):

            distances=np.sum(np.abs(self.Xtr - X[i, :]), axis=1)


 
            min_index=np.argmin(distances)

            Ypred[i]=self.Ytr[min_index]

        return Ypred




nn = NearestNeighbor() 

nn.train(Xtr_rows, Ytr)
Yte_predict = nn.predict(Xte_rows)  
print ('précision L1: %f' % ( np.mean(Yte_predict == Yte) ))