import pretty_midi
import numpy as np
import cPickle
import gzip
import glob

import matplotlib.pyplot as plt
from scipy.misc import imsave

fs = 10.0
note = (21,109)

def midi2pianoroll(dataset): 
    files = glob.glob(dataset)
    num = 0
    l = []
    for f in files:
        filename = f.replace('.mid','.png')
        pm = pretty_midi.PrettyMIDI(f)
        res = pm.get_piano_roll(fs=fs)
        res[np.where(res>0)] = 1
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

def piano_roll_to_pretty_midi(pr, program=1):
    piano_roll = np.zeros((128,pr.shape[1]))
    piano_roll[21:109,:] = pr
    print np.max(pr)
    
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


if __name__=="__main__":
    data = midi2pianoroll('../data/midi_set/*.mid')
    data = data.astype(np.int8)
    res = (data,np.ones(data.shape[0]))
    with gzip.open("../data/midi.pkl.gz",'wb') as f:
        cPickle.dump(res,f)
    

   
   
