import os
import sys
import timeit
import gzip
import pickle

import numpy as np
import cPickle

import theano
import theano.tensor as T


def data():
    num = np.random.randint(low=0,high=255,size=(10000,1),dtype=np.uint8)
    x =  np.unpackbits(num,axis=1)
    y = num >125

    
    return x

def som(x):
    np.random.uniform(0,1,size=(8,20))
    

    
if __name__ == "__main__":


    data()
