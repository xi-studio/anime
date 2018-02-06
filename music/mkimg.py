# -*- coding: UTF-8 -*-
import glob
import os
import librosa
import h5py
import numpy as np
from scipy import misc


def save(path):
    f = h5py.File(path,'r')
    spath = path.replace('h5','png').replace('m_img','song_img')
    print(spath)
    misc.imsave(spath,f['input'][:][0])
    f.close()
    
f = glob.glob('/data1/m_img/*.h5')

for x in f:
    save(x)
    

    
