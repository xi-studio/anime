from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


x = 1
w = np.arange(1,11)
res = np.random.uniform(0,1,size=10)
res = res/res.sum()
w = w/w.sum()
b = 0.1

def show(data,data1,dmax):
    plt.plot(data,data1)
    plt.ylim(0,dmax)
    plt.show()
    plt.clf()

for num in range(20000):
    #w = w + w*(x*w>b)
    w = w - 1*x*w*(x*w - res)
    w = w/w.sum()
    #plt.plot(np.arange(10),w,np.arange(10),res)
    #plt.show()

    print np.sum(np.abs(res-w))

    #w = w/w.sum()

