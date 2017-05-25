import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import matplotlib.pyplot as plt

class sfnn_layer(input, n_in, n_out):
    self.w = np.random.uniform(0,1,(784,50))
    self.n = np.ones(n_out)
    self.bmin = np.zeros(n_out)
    self.bmax = np.ones(n_out) * 3
    self.lr = np.ones(n_out) * 0.01
    self.grad = np.ones(n_out) * -1
    

L0 = np.ones(784)
W0 = np.random.uniform(0,1,(784,50))

L1 = np.ones(50)
F1 = -np.ones(50)
W1 = np.random.uniform(0,1,(50,50))

L2 = np.ones(50)
F2 = -np.ones(50)
W2 = np.random.uniform(0,1,(50,10))

L3 = np.ones(10)
F3 = -np.ones(10)


res = np.dot(L0,W0)
L1 = res*(res>0)
W0 = W0 - L0 * 

res = np.dot(L1,W1)
L2 = res*(res>0)

res = np.dot(L2,W2)
L3 = res*(res>0)





    


