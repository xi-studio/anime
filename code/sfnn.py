import numpy as np
import cPickle
import gzip
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize

from profilehooks import profile
import networkx as nx
import matplotlib.pyplot as plt


class network(object):
    def __init__(self, nsize=1000, w=None, b=0, lr=0):
        
        self.nsize = nsize

        if w!=None:
	    self.w = w    
        else:
	    self.w = csc_matrix((nsize,nsize),dtype=np.float32) 

        self.b  = b
        self.lr = lr
        self.v  = np.zeros(nsize,dtype=np.float32)

        self.idx = self.w.nonzero()

        self.head = None
        self.tail = None


    def totrain(self):
	idx = self.w.nonzero()
        row = np.arange(self.w.nnz)

        head = csc_matrix((np.ones(self.w.nnz), (row, idx[0])), shape=(self.w.nnz, self.nsize))
        tail = csc_matrix((np.ones(self.w.nnz), (row, idx[1])), shape=(self.w.nnz, self.nsize))
        
        self.head = head
        self.tail = tail.T


    def run(self,times=10):
        for n in range(times):
           if self.lr == 0:
	       self.step_predict()
	   else:
	       self.step_train()

           #print 'lr:',self.lr
	   #print self.v.sum()
	   #show((self.v.toarray())[0],3)
    
    def step_predict(self):
        res    = self.w.dot(self.v) 
        self.v = res * (res > self.b)

    def step_train(self):
        x = self.head.dot(self.v) * self.w.data
        self.v = self.tail.dot(x)

        self.w.data = self.w.data + self.lr * self.tail.T.dot(self.v) * x 
        self.w  = normalize(self.w, norm='l1', axis=0)


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
    w = csc_matrix((data, (idx[:,1], idx[:,0])), shape=(1000,1000))
    w = normalize(w, norm='l1', axis=0)

    return w

def show(data,dmax):
    plt.plot(data)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()
 

def sfnn_encode(data):
    filename = '../data/weight/mnist_size_3000.pkl'
    n = network(b=0.1,lr=0)
    n.load_w(filename)
    base = np.zeros((data.shape[0],1000*6))

    for num,x in enumerate(data):
        n.v[:,:784] = x 
        result = n.run(times=5)
        base[num] = result
        print(num)

    return base

def load_data(num):
    with gzip.open('../data/mnist.pkl.gz','rb') as f:
        data_set = cPickle.load(f)
    return  data_set[0][0][:num],data_set[0][1][:num]

def test():
    data,y = load_data(3000)
    filename = '../data/weight/mnist_size_3000.pkl'
    w = graph() 
    n = network(w=w, b=0.1, lr=0)

    base = np.zeros((data.shape[0],1000*5))
    for epoch in range(30):
        res = 0
        for num,x in enumerate(data):
            n.v[:,:784] = x 
            result = n.run(times=5)
            #result = n.run(times=5)
	    #base[num] = result
	    if num%100==0:
	        print num
                res = res + np.sum(np.abs(n.w - w))
                w = n.w
            
	print 'epoch',epoch 
        print 'res',res
	 
#        print 'epoch:',x
#        print 'sum:',n.v.sum()
#        #show(n.w.toarray()[:,100],0.3)
#
    n.save_w(filename)

@profile
def test_speed():
    filename = '../data/weight/work1.pkl'
    w = graph() 
    n = network(w=w, b=0.1, lr=1)
#    n = network(b=0.1,lr=0)
#    n.load_w(filename)

    pic = np.ones(700)
    n.totrain()
    for x in range(3000):
        n.v[:700] = pic 
	n.run(times=5)
#    n.save_w(filename)
    
    
if __name__=='__main__': 
    test_speed()
    


