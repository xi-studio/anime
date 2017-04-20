import numpy as np
from scipy.sparse import csc_matrix

import networkx as nx


class network(object):
    def __init__(self, w=None, nnum=1000, trainable=False):
        if w!=None:
	    self.w = w    
        else:
	    self.w = csc_matrix((nnum,nnum),dtype=np.float32) 
        self.v = csc_matrix((nnum,1),dtype=np.float32)
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
        res = self.w.dot(self.v)  
        self.v = res.multiply(res > 0)
	print np.sum(self.v)

    def step_train(self):
        v_stash = self.v
        res = self.w.dot(self.v)  
        self.v = res.multiply(res > 0)
        self.w = self.w - 0.01 * v_stash * self.v.T
	print np.sum(self.v)

def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

    
if __name__=='__main__':
    BA = nx.random_graphs.barabasi_albert_graph(1000, 1)
    idx = np.array(BA.edges())
    data = np.random.uniform(0,1,idx.shape[0])
    w = csc_matrix((data, (idx[:,0], idx[:,1])), shape=(1000,1000))

    n = network(w=w)
    n.v[:200] = np.random.uniform(0,10,(200,1))
    #n.trainable = True
    n.run(times=10)


