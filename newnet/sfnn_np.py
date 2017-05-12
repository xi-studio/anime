import numpy as np
import cPickle
import gzip
from scipy.sparse import *
from sklearn.preprocessing import normalize

from profilehooks import profile
import networkx as nx
import matplotlib.pyplot as plt


class network(object):
    def __init__(self, nsize=1000, w=None, b=None, e=None, dr=0.1, lr=0, bmax=2):
        
        self.nsize = nsize

        if w is None:
            self.w = np.zeros((self.nsize,self.nsize))
        else:
	    self.w = w    
	
	if b is None:
            #self.b = np.random.uniform(0,1,size=self.nsize)
            self.b = np.ones(self.nsize)*0.01
	else:
            self.b = b
	    
	if e is None:
            #self.e = np.ones(self.nsize)
            self.e = np.random.uniform(0,1,size=self.nsize)
	else:
            self.e = e

	self.dr = dr
        self.lr = lr
	self.bmax = bmax

        self.v  = np.zeros(nsize)
        self.v_cash = np.zeros((nsize,1))


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
        self.v_cash[:,0] = self.v
        res = np.dot(self.v,self.w)

	#self.v = (res - self.b) * (res > self.b) * (res <self.c) + self.c * (res >=self.c)
        self.v = (res - self.b) * ((res - self.b) > 0)

        self.w = self.w + self.lr * self.v_cash * self.w * (self.v * self.e)
        self.w = normalize(self.w, norm='l1', axis=1)
	#self.b = self.b + self.lr * res * self.dr * self.e
       

        #fig, ax = plt.subplots(figsize=(10, 10))
        #plt.imshow(self.w)
        #plt.show()

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
    filename = '../data/weight/test_200.pkl'
#    w = graph(nsize=200,edges=10) 
#    n = network(w=w, b=0.1, lr=1, nsize=200)
    n = network(b=0.1,lr=1, nsize=200)
    n.load_w(filename)

    pic = np.ones(50)
    for x in range(30):
        n.v[:50] = pic 
	n.run(times=10)
        #show(n.b,1)
#    n.save_w(filename)
    
    
if __name__=='__main__': 
    #plt.figure(num=None,figsize = (20,20))
    test()
    


