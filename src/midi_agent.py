import numpy as np

def E_cost(status,action):
    return np.sum(action > 0) + np.std(status - action)

def play(Energy, ticks )
    for n in arange(ticks):
        action = np.random.randint(low=0,high=2,size=127)


    
