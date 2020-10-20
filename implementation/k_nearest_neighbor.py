from __future__ import print_function
from data_utils import load_CIFAR10
import numpy as np
import random
import matplotlib.pyplot as plt
#from past.builtins import xrange
import sys
from collections import Counter

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

        
cifar10_dir = ('../datasets/cifar-10-batches-py/')
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):


        dists[i, j] = np.linalg.norm(self.X_train[j] - X[i])

    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):


      dists[i, :] = np.linalg.norm((self.X_train - X[i]), axis=-1)


    return dists

  def compute_distances_no_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 


    dists = np.sqrt(np.sum(X**2, axis=1).reshape(num_test, 1) + np.sum(self.X_train**2, axis=1) - 2 * X.dot(self.X_train.T))


    return dists

  def predict_labels(self, dists, k=1):

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):


      closest_y = []



      top_k_indx = np.argsort(dists[i])[:k]
      closest_y = self.y_train[top_k_indx]


      vote = Counter(closest_y)
      count = vote.most_common()
      y_pred[i] = count[0][0]



    return y_pred


best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# On calcule et affiche la précision
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('On obtient %d bonne estimation / %d estimation  => on a donc une précision de : %f' % (num_correct, num_test, accuracy))


