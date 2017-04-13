from scipy import misc
import numpy as np
import matplotlib.pyplot as plt


base = '../data/img/%i.npy'
data = np.zeros((784,60*(50+5)))
for x in range(60):
    res = np.load(base % (x))
    plt.plot(np.std(res,axis=1))
    plt.ylim(0,0.1)
    plt.savefig("../data/plot/plot_%i.png"%x)
    plt.clf()
    data[:,x*(50+5): (x+1)*(50+5) - 5] = res

print np.max(data)
np.save('../data/img/res.npy',data)
misc.imsave('../data/img/res.png',data)
