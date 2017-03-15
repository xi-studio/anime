import numpy as np

NEURON_NUM = 20
W_MAX_NUM  = 50 

active = np.zeros(W_MAX_NUM) 
head   = np.zeros(W_MAX_NUM) 
tail   = np.zeros(W_MAX_NUM) 
w      = np.zeros(W_MAX_NUM) 
fp_in  = np.zeros(W_MAX_NUM) 
fp_out = np.zeros(W_MAX_NUM) 
bp_in  = np.zeros(W_MAX_NUM) 
bp_out = np.zeros(W_MAX_NUM) 
nvalue = np.zeros(NEURON_NUM) 
ntype  = np.ones(NEURON_NUM) 
        
    
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    

def search_act():

def act_sum():


if __name__=='__main__':
    head[:8] = np.arange(8)
    head[8:16] = np.arange(8)
    tail[:8] = 8
    tail[8:16] = 9

    w[:16] = np.random.uniform(0,1,size=16)
    data = getdata()
    
    for num in np.arange(10):
	error = []
        for (num,x) in enumerate(data[0]):
	    fp_in[:8] = x
	    fp_in[8:16] = x
	    active[:16] = 1
            
            idx = np.where(active==1)
	    fp_out[idx] = fp_in[idx] * w[idx] 

	    nvalue[8] = np.sum(fp_out[:8])
	    nvalue[9] = np.sum(fp_out[8:16])
	    nvalue = nvalue * (nvalue>=0) 
            nvalue[nvalue>1] = 1

#	    cost= res - data[1][num]
#
#            idy = np.where(w[:,2]==8)
#	    w[:,6][idy] = cost
#	    w[:,7][idy] = w[:,6][idy] * w[:,4][idy]
#	    w[:,4][idy] = w[:,4][idy] - 0.01* w[:,3][idy] * w[:,6][idy]
#	    error.append(cost)
            print nvalue


