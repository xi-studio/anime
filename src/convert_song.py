from __future__ import division
from scipy.io import wavfile
import numpy as np
import cPickle
import gzip
import glob


rat = 11025
bit = np.power(2,16)
def conv2img():
    base = np.zeros((1,rat))
    res = glob.glob('../data/song/*.wav')
    for f in res:
        fs,x = wavfile.read(f)
        song = x[::4]/bit 
        idx = int(np.floor(song.shape[0]/rat))
        song = song[:rat*idx]
        song = song.reshape((idx,rat))
        base = np.append(base,song,axis=0)
    print base.shape
    print base.dtype
    return base
        

if __name__ == '__main__':
    res = conv2img()
    ta = np.ones(res.shape[0])
    res = res[:-(res.shape[0]%100)]
    print res.shape
    print res.dtype
    data = (res,ta)
   
    with gzip.open("../data/yiruma.pkl.gz",'wb') as f:
        cPickle.dump(data,f)
        

