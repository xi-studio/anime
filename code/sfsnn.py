import numpy as np
from scipy.sparse import csc_matrix

class network(object):
    def __init__(self, wmatrix=None, nmatrix=None, nnum=1000, wnum=5000, trainable=False):
        NEURON_NUM = nnum
        W_MAX_NUM  = wnum 
            
        self.w  = np.zeros(W_MAX_NUM,dtype=np.float32) 
        self.nv = np.zeros(NEURON_NUM,dtype=np.float32)
        self.ns = np.ones(NEURON_NUM,dtype=np.int8) 
        self.b  = np.zeros(NEURON_NUM,dtype=np.float32)

        if wmatrix==None:
            self.head = csc_matrix((W_MAX_NUM,NEURON_NUM),dtype=np.int8)
            self.tail = csc_matrix((W_MAX_NUM,NEURON_NUM),dtype=np.int8) 
        else:
            length = wmatrix.shape[0]
            row    = np.arange(lenth)
            data   = np.ones(length)
            col_head = wmatrix[:,0]
            col_tail = wmatrix[:,1]
            
            assert np.max(col_head) <= nnum
            assert np.max(col_tail) <= nnum

            self.head = csc_matrix((data,(row,col_head)),shape=(W_MAX_NUM,NEURON_NUM),dtype=np.int8)
            self.tail = csc_matrix((data,(row,col_tail)),shape=(W_MAX_NUM,NEURON_NUM),dtype=np.int8) 
            self.w[:length] = wmatrix[:,2]

        if nmatrix!=None:
            length = nmatrix.shape[0]
            self.nsymbol[:length] = nmatrix
          
        self.trainable = trainable    
        
    def step(self):
        active_w = self.head.dot(self.nv * self.ns)
        output   = (self.tail.T).dot(active_w)
        active_n = output * ((output - self.b)>=0)

        if self.trainable:
           lr = 0.01
           self.w = self.w + lr * self.tail.dot(output) * active_w
        
        
    
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

    
    
if __name__=='__main__':
    n = network()
    for x in range(1000):
        print x
        n.step()


