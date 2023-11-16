from labjack import ljm
import numpy as np 
import time
import matplotlib.pyplot as mpl
from signalfunctions import *
from scipy import signal

data = np.array([])

#Opening labjack and generating handle object
handle = ljm.openS("T7","ETHERNET","470010370")

#Writing constants for Stream-Out
ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",0)
ljm.eWriteName(handle,"STREAM_OUT0_TARGET",1000)
ljm.eWriteName(handle,"STREAM_OUT0_BUFFER_SIZE",2048)
ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",1)

size = 1

val = np.arange(2.5,5,2.5/(size*128))

#Writing the waveform onto the labjack
ljm.eWriteName(handle,"STREAM_OUT0_LOOP_SIZE",size*127)
for x in range(4):
	ljm.eWriteNameArray(handle,"STREAM_OUT0_BUFFER_F32",127,val[x*128:(x+1)*127])
ljm.eWriteName(handle,"STREAM_OUT0_SET_LOOP",1)

ljm.eStreamStart(handle,size*127,2,[4800,0],150)


for x in range(2):
	read = ljm.eStreamRead(handle)
	data = np.append(data,read[0][0:size*127])
	print x

ljm.eStreamStop(handle)
print("Stream finished")

theospace = np.linspace(size*127,2*size*127,size*127)
slope = ((2.5 * theospace)/(size*127))
space = np.linspace(2.5,5,size*127)
print('Voltage max: {0}'.format(data.max()))
data = np.convolve(data,signal.gaussian(size*127,10),'same')[(size*127)-1:(2*size*127)-1]
peakind = np.array(signal.argrelmax(np.array([x if x>0.99*data.max() else 0 for x in data])))
peak = np.zeros(size*127)+2.5
peak[peakind] = 5
if len(peakind)>0 :
	print(slope[peakind.mean()])
	ljm.eWriteName(handle,"DAC0",slope[peakind.mean()])
	ljm.close(handle)
	mpl.step(space,normalize_set(data,slope))
	mpl.step(space,slope)
mpl.step(space,peak)
mpl.xlabel('Applied voltage')
mpl.ylabel('Measured voltage')
mpl.show()