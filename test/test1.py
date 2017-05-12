import numpy as np
from scipy.sparse import csc_matrix

from profilehooks import profile


num = 1000

#data = np.random.choice(np.array([0,1],dtype=np.float64),size=(num,num),p=[0.9999,0.0001])
#row,col = np.where(data==1)
#print row
#print col
#print row.shape

#m = csc_matrix((np.ones(row.shape),(row,col)),shape=(num,num))


m = csc_matrix((np.ones(num),(np.arange(num),np.arange(num))),shape=(num,num))

m1 = csc_matrix((np.ones(1000),(np.arange(1000),np.arange(1000))),shape=(num,num))

x = np.ones(1000)


def test1():
    for n in range(10000):
        m.dot(x)
    print 'test1'

def test2():
    for n in range(10000):
	m1.dot(x)
    print 'test2'

@profile
def main():
    test1()
#    test2()

if __name__=='__main__':
    main()
