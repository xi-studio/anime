import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

head = np.random.randint(low=0,high=10,size=20)
tail = np.random.randint(low=0,high=10,size=20)

row = np.arange(20)
data = np.ones(20)
a = csc_matrix((data, (row,head)),shape=(20,10)).toarray()
b = csc_matrix((data, (row,tail)),shape=(20,10)).toarray()


def plotCM(cm,title,colorbarOn,givenAX):
    ax = givenAX
    idx = np.arange(10)
    idy = np.arange(20)
    plt.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=5.0)
    ax.set_xticks(range(10))
    ax.set_xticklabels(idx)


    plt.title(title,size=12)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,int(cm[i,j]),va='center', ha='center')


#fig1=plt.subplot(1, 3, 1)
#plotCM(a,"Head Index","off",fig1.axes)

fig2=plt.subplot(1, 1, 1)
w = np.random.randn(20,1)
plt.matshow(w, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
for x in range(20):
    fig2.axes.text(0,x,w[x,0],va='center', ha='center')

#fig3=plt.subplot(1, 3, 3)
#plotCM(b,"Tail Index","off",fig3.axes)

plt.show()

 
