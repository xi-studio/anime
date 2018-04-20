import glob
import pretty_midi
import h5py
from scipy import misc
import numpy as np

r = glob.glob('data/midi/*.mid')

FS = 16

num = 0
for x in r:
    try:
        p = pretty_midi.PrettyMIDI(x)
        img = p.get_piano_roll(fs=50)
        #misc.imsave('data/img/%s.png' % str(num),img[:,:1000])
        img = img.astype(np.uint8)
        #f = h5py.File(x.replace('midi','img'),'w')
        #f['data'] = img[12:96]
        #f['data'] = img
        #f.close()
        print np.sum(img[108:])
        num+=1
        #print num
        print x
    except Exception as e:
        print e
    
    

     
