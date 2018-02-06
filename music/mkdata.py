# -*- coding: UTF-8 -*-
import glob
import os
import librosa
import h5py
import numpy as np


def save(num,fs,d):
    path = '/data1/m_img/%d.h5' % num
    f = h5py.File(path,'w')
    input_  = d[:fs].reshape((1,512,512))
    output_ = d[fs:].reshape((1,512,512))
    f.create_dataset("input", data=input_)
    f.create_dataset("output", data=output_)
    f.close()
    
f = glob.glob('/data1/liminrun/*.mp3')

num = 0
fs = np.power(2,18)
for i,x in enumerate(f):
    print i
    data,_ = librosa.load(x)
    length = np.floor(data.shape[0]/fs)
    for j in range(int(length)-1):
        res = data[fs*j:fs*(j+2)] 
        save(num,fs,res)

        num = num + 1
    
