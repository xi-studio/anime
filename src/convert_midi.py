from __future__ import division
import midi
import numpy as np
import cPickle
import gzip
import glob

import matplotlib.pyplot as plt


pic_size = 2+37+11+34
print 'Pic size',pic_size

def note2array(onoff,pitch,velocity,tick):
    if tick > 960*2:
        tick = 960*2
    
    pitch = np.round(pitch / (127.0 / 36))
    velocity = np.round(velocity / (127.0 / 10))
    tick = np.round(tick / (960.0 / 16))
    
    pic = np.zeros(pic_size)
    pic[int(onoff)]   = 1
    pic[int(2+pitch)] = 1
    pic[int(2+36+velocity)] = 1
    pic[int(2+36+10+tick)]  = 1

    return pic 

def midi2key(filename,tracknum=0):
    p = midi.read_midifile(filename)
    events = p[tracknum]
    result = []

    for e in events:
        if isinstance(e,midi.NoteOnEvent):
            result.append(note2array(1,e.pitch,e.velocity,e.tick))
        if isinstance(e,midi.NoteOffEvent):
            result.append(note2array(0,e.pitch,e.velocity,e.tick))
            



def midi2matrix(dataset): 
    files = glob.glob(dataset)
    num = 0
    result = []
    for f in files:
        p = midi.read_midifile(f)
        events = p[0]

        for e in events:
            if isinstance(e,midi.NoteOnEvent):
                result.append(note2array(0,e.pitch,e.velocity,e.tick))
            if isinstance(e,midi.NoteOffEvent):
                result.append(note2array(0,e.pitch,e.velocity,e.tick))

    
        num = num + 1
       
    print num
    print len(result)
    return np.array(result)



if __name__=="__main__":

    data = midi2matrix('../data/midi_set/*.mid')
    size  = 6000
    idx = np.arange(data.shape[0]-84)
    np.random.shuffle(idx)
    idx = idx[:size]
    res = []
    for x in idx:
        res.append(data[x:x+84,:].reshape(84*84))
       
    res = np.array(res)
     
    data = res.astype(np.int8)
    ta = np.ones(data.shape[0])
    print data.shape
    res = (data,ta)
    with gzip.open("../data/midi.pkl.gz","wb") as f:
        cPickle.dump(res,f)

   
   
