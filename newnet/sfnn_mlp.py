import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import matplotlib.pyplot as plt


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
    g = np.ones(10)
    b = np.zeros(10)

    for iter in range(100):
        epo_sum = 0
        for num,x in enumerate(x_data):
            res = np.dot(x,w) + b
       	    res = res * (res>=0) 
            w  = w - 0.001 * x[:,None] *(res>0) * g #- np.sign(w)*0.0001
	    b  = b - 0.001 * g * (res>0)
            g  = res - y_data[num]
	    #print np.sum(np.abs(g))
            epo_sum += np.sum(np.abs(g))
	#print np.sum(np.abs(w))
        print epo_sum
    



    


    






    


