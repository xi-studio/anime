import numpy as np

class network(object):
  
    def __init__(self):
        NEURON_NUM = 1000
        W_MAX_NUM  = 2000 
            
        self.head    = np.zeros((W_MAX_NUM, NEURON_NUM),dtype=np.int8)
        self.tail    = np.zeros((W_MAX_NUM, NEURON_NUM),dtype=np.int8) 
        self.w       = np.zeros(W_MAX_NUM,dtype=np.float32) 
        self.nvalue  = np.zeros(NEURON_NUM,dtype=np.float32)
        self.nsymbol = np.ones(NEURON_NUM,dtype=np.int8) 
            
        
    def step(self):
        res = np.dot(np.dot(self.head,self.nvalue * self.nsymbol) * self.w, self.tail)
        res = res * (res>=0)
        print np.sum(res)
        
    
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

if __name__=='__main__':
    n = network()
    for x in range(100):
        n.step()


