import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import networkx as nx
import matplotlib.pyplot as plt

def show(data,dmax):
    plt.plot(data)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()


if __name__=='__main__':
    
    w = np.random.uniform(0,1,size=10)
    
    for x in range(10):
        w = w + w*(w>0.1)
        w = w/w.sum()
        show(w,1)
    
