from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize 

x = np.random.uniform(0,1,700)
print np.sum(x)
#x = np.random.uniform(0,1,10)
#x = np.arange(30)
#x = x/np.sum(x)
y = np.zeros(10)
y[0] = 0.321111
y[3] = 0.555

w = np.random.uniform(0,1,(700,10))
w = normalize(w, norm='l1', axis=1)

for n in range(30):
    res = np.dot(x,w)
    print res
    res[np.where(res>1)]= 1
    g = res - y 
    plt.plot(res)
    plt.ylim(-1,3)
    plt.show()
    w = w - x[:,None] * w * g
    print np.sum(w)
    #w = normalize(w, norm='l1', axis=1)
    #print np.sum(np.abs(g))
