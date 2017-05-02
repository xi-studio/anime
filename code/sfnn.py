import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
import cPickle

import networkx as nx
import matplotlib.pyplot as plt

class network(object):
    def __init__(self, nsize=1000, w=None, b=0, lr=0):
        
        self.nsize = 1000

        if w!=None:
	    self.w = w    
        else:
	    self.w = csc_matrix((nsize,nsize),dtype=np.float32) 

        self.b  = b
        self.lr = lr
        self.v  = csc_matrix((1,nsize),dtype=np.float32)

    def run(self,times=10):
        for n in range(times):
           if self.lr == 0:
	       self.step_predict()
	   else:
	       self.step_train()

           #print 'lr:',self.lr
	   #print self.v.sum()
	   #show((self.v.toarray())[0],1)
    
    def step_predict(self):
        res    = self.v.dot(self.w)  
        self.v = res.multiply(res > self.b)

    def step_train(self):
        v_stash = self.v
        res     = self.v.dot(self.w)  
        self.v  = res.multiply(res > self.b)

        self.w  = self.w + self.w.multiply(v_stash.T>0).multiply(self.v>0) * self.lr
        self.w  = normalize(self.w, norm='l1', axis=1)

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

def graph():
    #BA = nx.random_graphs.erdos_renyi_graph(1000, 0.2)
    BA = nx.random_graphs.barabasi_albert_graph(1000,10) 
    idx = np.array(BA.edges())
    data = np.random.uniform(0,1,idx.shape[0])
    #data = np.ones(idx.shape[0])
    w = csc_matrix((data, (idx[:,0], idx[:,1])), shape=(1000,1000))
    w = normalize(w, norm='l1', axis=1)

    return w


def show(data,dmax):
    plt.plot(data)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()
 
    
if __name__=='__main__':
    filename = '../data/weight/work1.pkl'
    #w = graph() 
    #n = network(w=w, b=0.05)
    n = network(b=0.1,lr=1)
    n.load_w(filename)
    #n.v[:,:20] = np.random.uniform(0,1,20)

    data = getdata()
    
    for x in range(20):
        for num in range(data[0].shape[0]):
            n.v[:,:8] = data[0][num]
	    n.v[:,8:9] = data[1][num]

            n.run(times=3)
        print 'epoch:',x
        print 'sum:',n.v.sum()
        #show(n.w.toarray()[:,100],0.3)

    #n.save_w(filename)
    


