import midi
import glob
path = '/Users/ZJY/icon/midi_set/*.mid'

r = glob.glob(path)

num = 0
for x in r:
    try:
       p = midi.read_midifile(x)
       if len(p)==3:
           p1 = midi.Pattern()
           p2 = midi.Pattern()
           p1.append(p[1])
           p2.append(p[2])
           midi.write_midifile("data/midi/%s_%s.mid" % (str(num),'1'), p1)
           midi.write_midifile("data/midi/%s_%s.mid" % (str(num),'2'), p2)
    
           print num
           num +=1
    except Exception as e:
        pass

print num

