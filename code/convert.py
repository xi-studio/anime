from __future__ import division
import numpy as np
import cPickle
import gzip
import midi
import glob

from midi.utils import midiread, midiwrite

r=(21, 109)
dt=0.3
scale = 400
scale_1d = scale


def align(data):
    res = []
    for (x,y) in data:
        if len(x) < scale_1d:
            base = np.zeros(2*scale_1d)
            base[ : len(x)] = x / (r[1]-r[0])
            base[scale_1d : len(y)+scale_1d] = y / scale_1d
            res.append(base)
        else:
            num = len(x) - scale_1d
            for idx in range(num):
                base = np.zeros(2*scale_1d)
                base[ : scale_1d] = x[idx : idx + scale_1d] / (r[1]-r[0])
                base[scale_1d : ] = y[idx : idx + scale_1d] / scale_1d
                res.append(base)
        
    print len(res)
    return np.array(res)
        
    
def miditomatrix(dataset): 
    files = glob.glob(dataset)
    base = []
    for f in files:
        print f
        data = midiread(f, r, dt).piano_roll
        if data.shape[0] <= scale:
            res = np.where(data>0)
            base.append(res)
        else:
            num = data.shape[0] - scale
            for idx in range(num):
                pic = data[idx : idx + scale]
                res = np.where(pic>0)
                base.append(res)

    res = align(base)
    return res


if __name__=="__main__":
    res = miditomatrix('../data_set/*.mid')
    with gzip.open('../data/400scale_seq.pkl.gz','wb') as f:
        cPickle.dump(res,f)
        print "save res"

    
    
