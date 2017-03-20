import numpy as np
from scipy.sparse import csc_matrix

class network(object):
  
    def __init__(self):
        NEURON_NUM = 10000
        W_MAX_NUM  = 100000 
            
        self.head    = csc_matrix((W_MAX_NUM,NEURON_NUM),dtype=np.int8)
        self.tail    = csc_matrix((W_MAX_NUM,NEURON_NUM),dtype=np.int8) 
        self.w       = np.zeros(W_MAX_NUM,dtype=np.float32) 
        self.nvalue  = np.zeros(NEURON_NUM,dtype=np.float32)
        self.nsymbol = np.ones(NEURON_NUM,dtype=np.int8) 
            
        
    def step(self):
        active_w = self.head.dot(self.nvalue * self.nsymbol)
        output = (self.tail.T).dot(active_w)
        output = output * (output>=0)

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
        n.step()


