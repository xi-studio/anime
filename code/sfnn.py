import numpy as np
from scipy.sparse import csc_matrix

class network(object):
    def __init__(self, nnum=1000, trainable=False):
            
        self.w = csc_matrix((nnum,nnum),dtype=np.float32) 
        self.b = np.zeros(nnum,dtype=np.float32)
        self.v = np.zeros(nnum,dtype=np.float32)
    
    def step(self):
        res = self.w.dot(self.v) + self.b 
	self.v = res * (res > 0)

	print self.v.sum()
    

    
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


