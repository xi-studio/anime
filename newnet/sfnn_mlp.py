import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import matplotlib.pyplot as plt

class w_layer(object):
    def __init__(self, input, n_in, n_out):
        self.w = np.random.uniform(0,1,(n_in,n_out))

class sfnn_layer(object):
    def __init__(self, num):
        self.v = np.zeros(num)
        self.b = np.zeros(num)
        self.g = np.ones(num)
        self.lr = 0.1
        
    






    


