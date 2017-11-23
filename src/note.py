import pretty_midi
import glob
import os
import numpy as np


def pm2matrix(pm):
    l = []
    for i,ins in enumerate(pm.instruments):
        for note in ins.notes:
            l.append((note.start, note.end, note.pitch, note.velocity, i)) 
    data = np.array(l,dtype = [('start', float), ('end', float), ('pitch', float),('velocity',float),('channel',float)])
    
    data = np.sort(data,order='start')
    
    matrix = np.zeros((len(l),5))
    matrix[:,0][1:] = data['start'][1:] - data['start'][:-1]
    matrix[:,1]     = data['end'] - data['start']
    matrix[:,2]     = data['pitch'] /128.0
    matrix[:,3]     = data['velocity'] /128.0
    matrix[:,4]     = data['channel']

    return matrix

def matrix2pm(matrix):
    tick = 0
    l = []
    for note in matrix:
        tick += note[0]
	start = tick
	end   = tick + note[1]
	pitch = int(np.floor(note[2] * 128))
	velocity = int(np.floor(note[3] * 128))
	channel = int(note[4])

	l.append((start,end,pitch,velocity,channel))
    
    data = np.array(l,dtype = [('start', float), ('end', float), ('pitch', int),('velocity',int),('channel',int)])
    data = np.sort(data,order='end')

    cMax = int(np.max(data['channel'])) + 1

    pm = pretty_midi.PrettyMIDI()

    for c in range(cMax):
        instrument = pretty_midi.Instrument(program=0)
	for note in data:
	    if note['channel'] == c:
                pm_note = pretty_midi.Note(
                            velocity=note['velocity'],
                            pitch=note['pitch'],
                            start=note['start'],
                            end=note['end'])
		instrument.notes.append(pm_note)
        pm.instruments.append(instrument)

    return pm

def getList():
    res = glob.glob("../data/piano/*/*/*/*.mid")
    for path in res:
	try:
            pm = pretty_midi.PrettyMIDI(path)
	    print path
	    print pm.instruments
	except Exception as e:
	    pass
    

if __name__ == '__main__':
    getList()




