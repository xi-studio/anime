from __future__ import division
import midi
import numpy as np
import cPickle
import gzip
import glob


SCALE = 28
PILEN = 1+25+1+1

def Note2array(onoff, pitch , velocity , tick ):
    pitch = int(pitch/5)
    note = np.zeros(PILEN)
    note[0] = onoff
    note[pitch+1] = 1 
    note[-2] = velocity/127.0
    note[-1] = tick/960.0

    return  note


def midi2keystrikes(filename,tracknum=0):
    p = midi.read_midifile(filename)
    events = p[tracknum]
    result = []
    
    for e in events:
        if isinstance(e,midi.NoteOnEvent):
            res = Note2array(1, e.pitch, e.velocity, e.tick)
            result.append(res)
        if isinstance(e,midi.NoteOffEvent):
            res = Note2array(0, e.pitch, e.velocity, e.tick)
            result.append(res)
            
    if (len(result) == 0) and (tracknum <5):
        # if it didn't work, scan another track.
        return midi2keystrikes(filename,tracknum+1)
        
    return np.array(result)   


def miditomatrix(dataset): 
    files = glob.glob(dataset)
    num = 0
    base = []
    for f in files:
        track = midi2keystrikes(f)
        size = int(np.ceil(track.shape[0]/SCALE))
        mod = track.shape[0] % SCALE
        if mod!=0:
            track = np.append(track,np.zeros((SCALE - mod,PILEN)),axis=0)

        for n in range(size):
            pic = track[SCALE*n:SCALE*(n+1),:]
            base.append(pic.reshape(SCALE * SCALE))
    
        num = num + 1
       
    print num
    return np.array(base)


if __name__=="__main__":
    data = miditomatrix('../data/midi_set/*.mid')
    data = data[:-(data.shape[0]%100)]
    target = np.ones(data.shape[0])
    res = ((data,target),(data,target),(data,target))
    with gzip.open("../data/midi.pkl.gz","wb") as f:
        cPickle.dump(res,f)

    
    