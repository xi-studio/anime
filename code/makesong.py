from __future__ import division
import numpy as np
import cPickle
import gzip
import midi
import glob
from midi.utils import midiread, midiwrite

r = (21,109)
dt = 0.3
scalse = 400

res = np.random.randint(low=0,high=2,size=(400,88))
filename = './one.mid'
midiwrite(filename, res, r, dt)
