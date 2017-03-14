import numpy as np

NEURON_NUM = 20
W_MAX_NUM = 10 

w = np.zeros((W_MAX_NUM,8)) # flag, node_head, node_tail, x_input, w, x_output
neuron = np.zeros((NEURON_NUM,2)) # value, type
        
    
def getdata():
    num = np.random.randint(low=0,high=255,size=(200,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125
    
    data = (x,y)
    return data    


if __name__=='__main__':
    w[:,2] = 8
    w[:8,1] = np.arange(8)
    w[:8,4] = np.random.uniform(0,1,size=8)
    data = getdata()

    for num in np.arange(10):
	error = []
        for (num,x) in enumerate(data[0]):
            w[:8,3] = x
	    w[:8,0] = 1
            idx = np.where(w[:,0]==1)

	    w[:,5][idx] = w[:,3][idx] * w[:,4][idx]

	    res = np.sum(w[:,5][idx])
	    res = res * (res>=0) 
	    if res>1:
                res = 1

	    cost= res - data[1][num]

            idy = np.where(w[:,2]==8)
	    w[:,6][idy] = cost
	    w[:,7][idy] = w[:,6][idy] * w[:,4][idy]
	    w[:,4][idy] = w[:,4][idy] - 0.01* w[:,3][idy] * w[:,6][idy]
	    error.append(cost)
        print np.sum(np.abs(np.array(error)))


