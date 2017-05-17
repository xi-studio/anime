from __future__ import division
from scipy.io import wavfile
import numpy as np
import cPickle
import gzip
import glob


rat = 11025
bit = np.power(2,16)
def conv2img():
    base = np.zeros((1,rat),dtype=np.int8)
    res = glob.glob('../data/song/*.wav')
    for f in res:
        fs,x = wavfile.read(f)
        song = x[::4]/bit
        idx = int(np.floor(song.shape[0]/rat))
        song = song[:rat*idx]/bit
        song = song.reshape((idx,rat))
        song = song.astype(np.int8)
        base = np.append(base,song,axis=0)
    print base.shape
    print base.dtype
    return base
        

if __name__ == '__main__':
    res = conv2img()
    with gzip.open("../data/song/yiruma.pkl.gz",'wb') as f:
        cPickle.dump(res,f)

