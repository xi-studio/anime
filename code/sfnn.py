import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize
import cPickle

import networkx as nx
import matplotlib.pyplot as plt

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
	#show((self.v.toarray())[0])

    def step_train(self):
        v_stash = self.v
        res = self.v.dot(self.w)  
        self.v = res.multiply(res > self.b)
        self.w = self.w + self.w.multiply(v_stash.T>0).multiply(self.v>0)
        self.w = normalize(self.w, norm='l1', axis=1)

	print (self.v>0).sum()
	#show((self.v.toarray())[0])

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


def show(data):
    plt.plot(data)
    plt.ylim(0,0.3)
    plt.show()
    plt.clf()
 
    
if __name__=='__main__':
    filename = '../data/weight/ba.pkl'
    #w = graph() 
    #n = network(w=w, b=0.01)
    n = network(b=0.1)
    n.load_w(filename)
    #n.v[:,:20] = np.random.uniform(0,1,20)
    for x in range(10):
        n.v[:,:200] = np.random.uniform(0,0.5,200)
        #n.v[:,:50] = np.arange(50)
        n.trainable = True
        n.run(times=10)
	show(n.w.toarray()[0])
    #n.save_w(filename)
    


