import numpy as np
from scipy.sparse import csc_matrix

class network(object):
    def __init__(self, nnum=1000, trainable=False):
            
        self.w = csc_matrix((nnum,nnum),dtype=np.float32) 
        self.v = csc_matrix((nnum,1),dtype=np.float32)
	self.trainable = trainable
    
    def step(self):
        v_stash = self.v
        res = self.w.dot(self.v)  
        self.v = res > 0

	if self.trainable == True:
	    self.w = self.w - 0.01 * v_stash * self.v.T

    
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


