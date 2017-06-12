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
        

def load_data():
    filepath = '../data/mnist.pkl.gz'
    f = gzip.open(filepath)
    train_data,test_data,valid_data = cPickle.load(f)
    x_data,y_data = train_data
    return x_data[:500],y_data[:500]

if __name__ =='__main__':
    x_data,idy = load_data()
    y_data = np.zeros((500,10))
    y_data[np.arange(500),idy] = 1

    w = np.random.uniform(0,1,(784,10))
    w = normalize(w, norm='l1', axis=1)

    g = np.ones(10)

   
    x = np.random.uniform(0,1,784)
    y = np.zeros(10)
    y[1] = 1

    print x
    for iter in range(1): 
        res = np.dot(x,w)
	res = res*(res>=0)
	w = w -  0.01*x[:,None] * w * (res>0) * g
	g = res - y 
	print g
    #    print g

#    for iter in range(100):
#        epo_sum = 0
#        for num,x in enumerate(x_data):
#            res = np.dot(x,w)
#       	    res = res*(res>=0) 
#            w = w -  0.01*x[:,None] * w * (res>0) * g 
#	   # w = w*(w>0)
#            g = res - y_data[num]
#           # w = normalize(w, norm='l1', axis=1)
#            epo_sum += np.sum(np.abs(g))
#	print epo_sum
#    
#


    


    






    


