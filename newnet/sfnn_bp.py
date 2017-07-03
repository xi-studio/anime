import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import networkx as nx
import matplotlib.pyplot as plt


class network(object):
    def __init__(self, nsize=1000, w=None, b=None, lr=0):
        
        self.nsize = nsize

        if w!=None:
	    self.w = w    
        else:
	    self.w = np.random.randn(nsize,nsize) 

        if b!=None:
	    self.b = b    
        else:
	    self.b = np.random.randn(nsize,nsize) 

        self.lr = lr
        self.v  = np.zeros(nsize)
	self.g  = 1 * np.ones(nsize) 


    def run(self,times=10):
        for n in range(times):
            res = np.dot(self.v,self.w) + self.b
	    activated = res>=0
            if self.lr != 0:
                self.w = self.w - self.lr * self.v * activated * self.g
	        self.b = self.b - self.lr * activated * self.g
	    self.v = res * activated 
	    print self.v.sum()

    def cost(self):
        print 'cost'

    def bp(self):
        w = self.w.T
	self.g = np.dot(self.g, w)


    def save_w(self,name):
	with open(name, 'wb') as f:
	    cPickle.dump(self.w,f)

    def load_w(self,name):
	with open(name, 'rb') as f:
	    self.w = cPickle.load(f)
        
        
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

def graph(nsize=1000,edges=10):
    #BA = nx.random_graphs.erdos_renyi_graph(1000, 0.2)
    BA = nx.random_graphs.barabasi_albert_graph(nsize,edges) 
    idx = np.array(BA.edges())
    data = np.random.uniform(0,1,idx.shape[0])


def show(data,dmax):
    plt.plot(data)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()
 

def load_data(num):
    with gzip.open('../data/mnist.pkl.gz','rb') as f:
        data_set = cPickle.load(f)
    return  data_set[0][0][:num],data_set[0][1][:num]


def test_speed():
    filename = '../data/weight/test_speed.pkl'
    w = graph(nsize=1000,edges=10) 
    n = network(lr=0.001, nsize=100)
#    n = network(b=0.1,lr=0)
#    n.load_w(filename)

    pic = np.ones(100)
    for x in range(30):
        #n.v[:700] = pic 
	n.run(times=5)
#    n.save_w(filename)
    
    
if __name__=='__main__': 
    test_speed()
    


