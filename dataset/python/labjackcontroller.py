from labjack import ljm
import numpy as np 
import matplotlib.pyplot as mpl
from scipy import signal
import argparse

#External function files
from signalfunctions import *

parser = argparse.ArgumentParser(prog='Labjack T7 470010370 controller debug version',description='''Outputs a sine wave of a given frequency that is used to control the Optotune Lens Driver 4. The sine wave is designed to output 2.5-4V on DAC0. This version is for debugging purposes and needs Scipy to run. You will also need drivers for labjack T7 and LJM interpreter.''',epilog='''Version 0.1''')
parser.add_argument('-v',"--verbosity",help ='Increase verbosity',default=0, action="count")
parser.add_argument('-fq',type=int,help='Frequency of sine wave in Hz. Defaults to 120 Hz',nargs='?',default=60)
parser.add_argument('-wvfrmres',type=int,help='Periods per 127 dots. Defaults to 2.',nargs='?',default=2)
parser.add_argument('-scantime',type=float,help='Waveforms scanned by each read function. Proportionnal to size of data packet sent. Defaults to 2.',default=2)
parser.add_argument('-acqtime',type=int,help='Number of times the read function is called.',default=50)
parser.add_argument('-compar', help ='Comparison between AIN0 and AIN1.', action = 'store_true', default=True)
parser.add_argument('-amp',help='Amplitude of oscillation',nargs='?',default=0.3)
parser.add_argument('-offset',help='Voltage around where the oscillation takes place.')
parser.add_argument('-calibration', help ='Allows calibration of offset voltage', action = 'store_false', default=True)
args = parser.parse_args()



if int(args.fq/float(args.wvfrmres)) != args.fq/float(args.wvfrmres):
	raise AssertionError('Waveform resolution not multiple of frequency')

#Waveform generation
fqreal = int(args.fq/float(args.wvfrmres))
sp = np.linspace(0, int(args.wvfrmres) * 2 * np.pi,128)
val = (args.amp*np.sin(sp))+3.7080

#Paramters for acquisition
scanres = int(args.scantime * int(127))
scanrate = fqreal*127

data1 = np.array([])
data2 = np.array([])

#Opening labjack and generating handle object
handle = ljm.openS("T7","ETHERNET","470010370")

#Writing constants for Stream-Out
ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",0)
ljm.eWriteName(handle,"STREAM_OUT0_TARGET",1000)
ljm.eWriteName(handle,"STREAM_OUT0_BUFFER_SIZE",256)
ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",1)

#Writing the waveform onto the labjack
ljm.eWriteName(handle,"STREAM_OUT0_LOOP_SIZE",127)
ljm.eWriteNameArray(handle,"STREAM_OUT0_BUFFER_F32",127,val[0:127])
ljm.eWriteName(handle,"STREAM_OUT0_SET_LOOP",1)

#Start streaming. 4800 is Stream-Out. 0 is Stream-In
ljm.eStreamStart(handle,scanres,3,[4800,0,2],fqreal*127)

#Mainloop for streaming. Could be a "while" loop for infinite looping, or until break happens
for i in range(args.acqtime):
	read = ljm.eStreamRead(handle)
	if args.compar == True:
		if i > 0:
			data1 = np.append(data1,read[0][0:2*scanres:2])
			data2 = np.append(data2,read[0][1:2*scanres:2])
	if args.verbosity > 0:
		print('DeviceBufferStatus: {0}  LJMBufferStatus:{1}'.format(read[1],read[2]))
	'''
	data2now = np.convolve(read[0][1:2*scanres:2],signal.gaussian(127/float(args.wvfrmres),2),'valid')
	data2nowsmooth = np.array([np.mean([data2now[y + (x*127/args.wvfrmres)] for x in range(args.scantime*args.wvfrmres -1)]) for y in range(127/args.wvfrmres)])
	data2nowsmoothmask = data2nowsmooth > 0.4*max(data2nowsmooth)
	data2nowsmoothpeaks = np.array([(data2nowsmooth)[x] if data2nowsmoothmask[x]==True else 0 for x in range(len(data2nowsmooth))])
	peakind = signal.argrelmax(data2nowsmoothpeaks)
	if len(peakind[0])==2:
		print('Ratio: {0}'.format((peakind[0][1]-peakind[0][0])*(args.wvfrmres*2)/127.0))
		table = np.append(table,(peakind[0][1]-peakind[0][0])*(args.wvfrmres*2)/127.0)
	else:
		table = np.append(table,0)
		print('NaN')
	'''


ljm.eStreamStop(handle)
ljm.close(handle)
print('Stream Finished')



if args.compar == True:
	
	plotspace = np.linspace(0,((args.acqtime-1)*args.scantime)/(float(fqreal)),scanres*(args.acqtime-1))
	plotspace2 = np.linspace(0,(1)/(float(args.fq)),127/args.wvfrmres)
	fig = mpl.figure()


	data3 = [1 if x>=np.mean(data1) else 0 for x in data1]
	spectrum = np.fft.rfft(data2)
	fftfreq = np.fft.rfftfreq(len(data2))
	data2norm = normalize_set(data2,data3)
	data2 = np.convolve(data2,signal.gaussian(127/float(args.wvfrmres),3),'valid')
	data2smooth = np.array([np.mean([data2[y + (x*127/args.wvfrmres)+127] for x in range(args.scantime*args.wvfrmres -1)]) for y in range(127/args.wvfrmres)])

	data2smoothmask = data2smooth > 0.5*max(data2smooth)
	data2smoothpeaks = np.array([(data2smooth)[x] if data2smoothmask[x]==True else 0 for x in range(len(data2smooth))])
	peakind = signal.argrelmax(data2smoothpeaks)
	if len(peakind[0]) == 2:
		a = (3*127)/(args.wvfrmres*4)
		data2shift = ((1*127)/(4*args.wvfrmres) - (peakind[0][0]+(peakind[0][1]-peakind[0][0])/(float(2))))*1/(float(fqreal * 127))
	else:
		print("Shift not calculated")
		data2shift = 0

	data4 = [1 if x in peakind[0] else 0 for x in range(len(data2smooth))]
	
	#fig.add_subplot(311)
	#Plot signals
	mpl.step(plotspace, normalize_set(data1,data3)+0.5)

	mpl.plot(plotspace2 + data2shift +(1)/(float(args.fq)),normalize_set(data2smooth,data3)+0.8)

	mpl.step(plotspace + data2shift +(1)/(2*float(args.fq)), data2norm+0.2)

	mpl.step(plotspace2+data2shift+(1)/float(args.fq),data4)
	mpl.ylim(-0.5,1.5)
	mpl.xlim((1)/(float(1.2*args.fq)), (2)/(float(0.8*args.fq)))
	#mpl.plot(plotspace, data2)
	mpl.show()

	spectrum = np.fft.fft(data2norm)
	freqseq = np.fft.fftfreq(len(spectrum[1:]),1/float(scanrate))
	#mpl.plot(freqseq,abs(spectrum)[1:])
	spectrum[np.where(np.logical_or(np.less(abs(freqseq),15),np.greater(abs(freqseq),100)))]=0
	data2mod = np.real(np.fft.ifft(spectrum))
	mpl.plot(freqseq,abs(spectrum)[1:])
	print('Dominant frequencies:{0}'.format(find_dominantfq(freqseq,spectrum,scanrate)))
	mpl.show()
	mpl.plot(plotspace,data3)
	mpl.plot(plotspace,normalize_set(data2norm,data3)+0.5)
	mpl.show()