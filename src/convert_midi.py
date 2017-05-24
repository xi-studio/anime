import pretty_midi
import numpy as np
import cPickle
import gzip
import glob

import matplotlib.pyplot as plt
from scipy.misc import imsave

fs = 10
note = (21,109)

def midi2pianoroll(dataset): 
    files = glob.glob(dataset)
    num = 0
    l = []
    for f in files:
        filename = f.replace('.mid','.png')
        pm = pretty_midi.PrettyMIDI(f)
        res = pm.get_piano_roll(fs=fs)
        data = res[21:109,:]
        data = data.T
        for x in range(data.shape[0]-100):
            l.append(data[x:x+100].reshape(-1))

        print f
    idx = np.arange(len(l))
    l = np.array(l)
    res = l[:10000]
    print l.shape
    print l.dtype
   
    return res


if __name__=="__main__":
    data = midi2pianoroll('../data/midi_set/*.mid')
    data = data.astype(np.int8)
    res = (data,np.ones(data.shape[0]))
    with gzip.open("../data/midi.pkl.gz",'wb') as f:
        cPickle.dump(res,f)
    

   
   
