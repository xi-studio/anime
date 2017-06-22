import numpy
import numpy as np
import theano
import theano.tensor as T
rng=numpy.random

N=400
feats=784
# D[0]:generate rand numbers of size N,element between (0,1)
# D[1]:generate rand int number of size N,0 or 1
D=(rng.randn(N,feats),rng.randint(size=N,low=0,high=2))
training_steps=10000

# declare symbolic variables
x=T.matrix('x')
y=T.vector('y')
w=theano.shared(rng.randn(feats),name='w')  # w is shared for every input
b=theano.shared(0.,name='b') # b is shared too.

print('Initial model:')
print(w.get_value())
print(b.get_value())

# construct theano expressions,symbolic
#p_1=1/(1+T.exp(-T.dot(x,w)-b))  # sigmoid function,probability of target being 1
#prediction=p_1>0.5
#xent=-y*T.log(p_1)-(1-y)*T.log(1-p_1)  # cross entropy
#cost=xent.mean()+0.01*(w**2).sum() # cost function to update parameters
#gw,gb=T.grad(cost,[w,b])  # stochastic gradient descending algorithm

res = T.dot(x,w)+b
res = res * (res>0)
prediction = res>0
xent=-y*T.log(res)-(1-y)*T.log(1-res)
cost=xent.mean()+0.01*(w**2).sum()
#xent = res - y
#cost = xent.mean()
gw,gb=T.grad(cost,[w,b])

##compile
train=theano.function(inputs=[x,y],outputs=[prediction,xent],updates=((w,w-0.1*gw),(b,b-0.1*gb)))
predict=theano.function(inputs=[x],outputs=prediction)

# train
for i in range(training_steps):
	    pred,err=train(D[0],D[1])

#	    print('Final model:')
#	    print(w.get_value())
#	    print(b.get_value())
#	    print('target values for D:')
#	    print(D[1])
#	    print('prediction on D:')
#	    print(predict(D[0]))

	    print('result',np.sum(predict(D[0]) - D[1]))

	    print('newly generated data for test:')
	    test_input=rng.randn(30,feats)
	    print('result:')
	    print(predict(test_input))
