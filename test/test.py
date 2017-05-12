from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import math

def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

foo = np.vectorize(is_prime)

model = Sequential()
model.add(Dense(500, input_dim=8, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(500, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(50, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(50, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


scale = 250.0
data = np.arange(int(scale),dtype=np.uint8)
#labels = data%2 
labels = foo(data)
data = np.unpackbits(data.reshape((int(scale),1)),axis=1)

index = np.arange(int(scale))
np.random.shuffle(index)

data = data[index]
labels = labels[index]
print data
print labels


num = 100
X_train = data[:num]
Y_train = labels[:num]
X_test = data[num:]
Y_test = labels[num:]

history = model.fit(X_train, Y_train, nb_epoch=30, batch_size=10)
score = model.evaluate(X_test,Y_test, batch_size=10)

print history.history['loss']

print "score",score

