import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize

import networkx as nx

class network(object):
    def __init__(self, w=None, b=0, nnum=1000, trainable=False):
        if w!=None:
	    self.w = w    
        else:
	    self.w = csc_matrix((nnum,nnum),dtype=np.float32) 

        self.b = b
        self.v = csc_matrix((1,nnum),dtype=np.float32)
	self.trainable = trainable

    def run(self,times=10):
        if self.trainable == True:
	    print "Train"
	    for n in range(times):
	        self.step_train()
	else:
	    print "Predict"
	    for n in range(times):
	        self.step_predict()
    
    def step_predict(self):
        res = self.v.dot(self.w)  
        self.v = res.multiply(res > self.b)

	print self.v.sum()

    def step_train(self):
        v_stash = self.v
        res = self.v.dot(self.w)  
        self.v = res.multiply(res > self.b)
        self.w = self.w + self.w.multiply(v_stash.T>0).multiply(self.v>0)
        self.w = normalize(self.w, norm='l1', axis=1)

	print self.v.sum()

def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

def graph():
    BA = nx.random_graphs.barabasi_albert_graph(1000,1) 
    idx = np.array(BA.edges())

    np.save('../data/ba_network.npy',idx)
    print 'save ok'
 
    
if __name__=='__main__':
#    graph()
    idx = np.load('../data/ba_network.npy')
    #BA = nx.random_graphs.barabasi_albert_graph(1000,20) 
    #idx = np.array(BA.edges())
    data = np.random.uniform(0,1,idx.shape[0])
    #data = np.ones(idx.shape[0])
    w = csc_matrix((data, (idx[:,0], idx[:,1])), shape=(1000,1000))
    w = normalize(w, norm='l1', axis=1)

    n = network(w=w, b=0.0)
    #n.v[:,:20] = np.random.uniform(0,1,20)
    for x in range(10):
        #n.v[:,:200] = np.random.randint(low=0,high=10,size=200)
        n.v[:,:200] = np.ones(200)
        n.trainable = True
        n.run(times=10)


