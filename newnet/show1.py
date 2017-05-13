import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


num = 500
BA = nx.random_graphs.barabasi_albert_graph(num, 5)

img =np.zeros((num,num))

idx = np.array(BA.edges())

img[idx[:,0],idx[:,1]]=1

#plt.plot(np.sum(img,axis=1)/2)
plt.matshow(img, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
plt.show()
