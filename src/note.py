import pretty_midi
import numpy as np

path = '../data/jiao.mid'

pm = pretty_midi.PrettyMIDI(path)


def pm2matrix(pm):
    l = []
    for i,ins in enumerate(pm.instruments):
        for note in ins.notes:
            l.append((note.start, note.end, note.pitch, note.velocity, i)) 
    data = np.array(l,dtype = [('start', float), ('end', float), ('pitch', float),('velocity',float),('channel',float)])
    print data
    
    data = np.sort(data,order='start')

    
    base = np.zeros((len(l),5))
    base[:,0][1:] = data['start'][1:] - data['start'][:-1]
    base[:,1]     = data['end'] - data['start']
    base[:,2]     = data['pitch'] /128.0
    base[:,3]     = data['velocity'] /128.0
    base[:,4]     = data['channel']

    return base

def matrix2pm(matrix):
    tick = 0
    l = []
    for note in matrix:
        tick += note[0]
	start = tick
	end   = tick + note[1]
	pitch = np.floor(note[2] * 128)
	velocity = np.floor(note[3] * 128)
	channel = note[4]

	l.append((start,end,pitch,velocity,channel))
    
    data = np.array(l,dtype = [('start', float), ('end', float), ('pitch', float),('velocity',float),('channel',float)])
    data = np.sort(data,order='end')

    print data
         

if __name__ == '__main__':
    r =  pm2matrix(pm)
    matrix2pm(r)        



