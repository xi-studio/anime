import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import networkx as nx
import matplotlib.pyplot as plt


class network(object):
    def __init__(self, nsize=1000, w=None, b=0, c=2, lr=0):
        
        self.nsize = nsize

        if w!=None:
	    self.w = w    
        else:
            self.w = np.zeros((nsize,nsize))

        self.b  = b
        self.c  = c
        self.lr = lr
        self.v  = np.zeros(nsize)
        self.v1 = np.zeros((nsize,1))


    def run(self,times=10):
        for n in range(times):
           if self.lr == 0:
	       self.step_predict()
	   else:
	       self.step_train()
           print self.v.sum()
    
    def step_predict(self):
        res = np.dot(self.v,self.w)
        self.v = res * (res > self.b)

    def step_train(self):
        self.v1[:,0] = self.v
        res = np.dot(self.v,self.w)
	self.v = res * (res > self.b) * (res <self.c)
        #self.v = res * (res > self.b)

        self.w = self.w + self.lr*self.v1*self.w* self.v
        self.w = normalize(self.w, norm='l1', axis=1)
       

        fig, ax = plt.subplots(figsize=(15, 15))
        plt.imshow(self.w)
        #plt.matshow(self.w, fignum=False, cmap='Blues', vmin=0, vmax=1.0 , aspect='auto')
        plt.show()
        #plt.clf()

    def save_w(self,name):
	with open(name, 'wb') as f:
	    cPickle.dump(self.w,f)

    def load_w(self,name):
	with open(name, 'rb') as f:
	    self.w = cPickle.load(f)
        

def graph(nsize=1000,edges=10):
    w = np.zeros((nsize,nsize))
    BA = nx.random_graphs.barabasi_albert_graph(nsize,edges) 
    idx = np.array(BA.edges())
    data = np.random.uniform(0,1,idx.shape[0])
    w[idx[:,0],idx[:,1]] = data
    w = normalize(w, norm='l1', axis=0)
    print w

    return w

def show(data,dmax):
    plt.plot(data)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()
 

def test():
    filename = '../data/weight/test_100.pkl'
    w = graph(nsize=100,edges=10) 
    n = network(w=w, b=0.01, lr=1, nsize=100)
#    n = network(b=0.1,lr=0)
#    n.load_w(filename)

    pic = np.ones(30)
    for x in range(20):
        n.v[:30] = pic 
	n.run(times=5)
#    n.save_w(filename)
    
    
if __name__=='__main__': 
    #plt.figure(num=None,figsize = (20,20))
    test()
    


