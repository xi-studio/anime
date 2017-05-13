from __future__ import division

import numpy as np
import cPickle
import gzip
import midi
import glob

from midi.utils import midiread, midiwrite

r=(21, 109)
dt=0.3
scale = 88

    
def miditomatrix(dataset): 
    genres = {'jigs':0,
              'morris':1,
              'reels':2, 
              'ashover':3, 
              'waltzes':4, 
              'hpps':5, 
              'slip':6, 
              'playford':7, 
              'xmas':8,
             }

    files = glob.glob(dataset)
    res_x = np.zeros((1,scale*(r[1]-r[0])),dtype=np.uint8)
    res_y = []
    for f in files:
        print f
        data = midiread(f, r, dt).piano_roll
        name = f.split('/')[-1]
        name = name.split('_simple_chords_')[0]
        if data.shape[0] <= scale:
            base = np.zeros((scale,r[1]-r[0]))
            base[:data.shape[0]] = data
            res_x = np.append(res_x, base.reshape((1,base.shape[0] * base.shape[1])), axis=0)
            res_y.append(genres[name])
        else:
            num = np.floor(data.shape[0] /scale)
            for idx in range(int(num)):
                pic = data[idx * scale : (idx + 1) * scale]
                res_x = np.append(res_x, pic.reshape((1,pic.shape[0] * pic.shape[1])), axis=0)
                #res_y.append(genres[name])

#    return (res_x,np.array(res_y,dtype=np.uint8)) 
    return res_x 

def get_genres():
   files = glob.glob('../data/data_set/*.mid')
   genres = []
   for f in files:
       res = f.split('/')[-1]
       res = res.split('_simple_chords_')[0]
       if res not in genres:
          genres.append(res)
   print genres

if __name__=="__main__":
    res = miditomatrix('../data/data_set/*.mid')
    with gzip.open('../data/88scale_data.pkl.gz','wb') as f:
        cPickle.dump(res,f)
        print "save res"

    
    
