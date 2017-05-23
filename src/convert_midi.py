from __future__ import division
import midi
import numpy as np
import cPickle
import gzip
import glob

import matplotlib.pyplot as plt

tmp = 960    # ticks per beat
ts  = tmp/16 # time scale
ps  = 3      # picth scale
sc  = 5     # 10 seconds

size = (16*sc, 40*2)
#size = (16*sc, (127//ps)*2)

print 'Pic size',size

def midi2key(filename,tracknum=0):
    p = midi.read_midifile(filename)
    events = p[tracknum]
    result = []

    abs_time = 0
    for e in events:
        if isinstance(e,midi.NoteOnEvent):
            abs_time += e.tick
            result.append(abs_time)
        if isinstance(e,midi.NoteOffEvent):
            abs_time += e.tick
            result.append(abs_time)
            
    if len(result)!=0:
        print abs_time/len(result)


def midi2keystrikes(filename,tracknum=0):
    p = midi.read_midifile(filename)
    events = p[tracknum]
    result = []

    abs_time = 0
    for e in events:
        if isinstance(e,midi.NoteOnEvent):
            if e.pitch >120:
                e.pitch = 120
            abs_time = abs_time + e.tick
            res = [int(abs_time / ts), (e.pitch // ps) * 2 + 1]
            result.append(res)
        if isinstance(e,midi.NoteOffEvent):
            abs_time = abs_time + e.tick
            res = [int(abs_time / ts), (e.pitch // ps) * 2]
            result.append(res)
            
    if (len(result) == 0) and (tracknum <5):
        return midi2keystrikes(filename,tracknum+1)

    timesize = result[-1][0]    
    timesize = np.ceil(timesize / (16*sc)) * 16 *sc
    base = np.zeros((int(timesize),40*2))
    idx = np.array(result)   
    base[idx[:,0],idx[:,1]] = 1
    base = base.reshape((-1,80*40*2))
    return base 


def midi2matrix(dataset): 
    files = glob.glob(dataset)
    num = 0
    base = []
    for f in files:
        track = midi2keystrikes(f)

        for x in track:
            base.append(x)
    
        num = num + 1
       
    print num
    return np.array(base)



if __name__=="__main__":
    dataset = '../data/midi_set/*.mid'
    files = glob.glob(dataset)
    for f in files:
        track = midi2key(f)

#    data = midi2matrix('../data/midi_set/*.mid')
#    data = data[:-(data.shape[0]%100)]
#    data = data.astype(np.int8)
#    ta = np.ones(data.shape[0])
#    print data.shape
#    res = (data,ta)
#    with gzip.open("../data/midi.pkl.gz","wb") as f:
#        cPickle.dump(res,f)

    
    
