import numpy as np

NEURON_NUM = 20
W_MAX_NUM  = 50 

head   = np.zeros((W_MAX_NUM, NEURON_NUM)) 
tail   = np.zeros((W_MAX_NUM, NEURON_NUM)) 
w      = np.zeros(W_MAX_NUM) 
nvalue = np.zeros(NEURON_NUM)
ntype  = np.ones(NEURON_NUM) 
        
    
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    


def onestep():
    res = np.dot(np.dot(head,nvalue * ntype) * w, tail)
    nvalue = res * (res>=0)
    

if __name__=='__main__':


