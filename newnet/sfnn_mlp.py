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
    






    


