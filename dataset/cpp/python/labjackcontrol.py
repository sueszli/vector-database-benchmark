from labjack import ljm
import numpy as np
import time
import matplotlib.pyplot as mpl
from scipy import signal
import traceback
import sys
import argparse

#External function files
from signal_processing_functions import *
from calibration_functions import *

parser = argparse.ArgumentParser(prog='Labjack T7 470010370 controller debug version',description='''Outputs a sine wave of a given frequency that is used to control the Optotune Lens Driver 4. Needs scipy to run for spike detection. You will also need drivers for labjack T7 and LJM interpreter. The program can always be broken with a keyboard interrupt (ctrl-c)''',epilog='''Version 1.0''')
parser.add_argument('-v',"--verbosity",help ='Increase verbosity',default=0, action="count")
parser.add_argument('-fq',type=int,help='Frequency of sine wave in Hz. Defaults to 90 Hz',nargs='?',default=90)
parser.add_argument('-wvfrmres',type=int,help='Periods per 127 dots. Defaults to 1.',nargs='?',default=1)
parser.add_argument('-readsize',type=int,help='Waveforms scanned by each read function. Proportionnal to size of data packet sent. Defaults to 3.',default=3)
parser.add_argument('-readnum',type=float,help='Number of times the read function is called. Is infinite by default',default=float('inf'))
parser.add_argument('-amp',type=float,help='Amplitude of oscillation. Defaults to 0.5',nargs='?',default=0.5)
parser.add_argument('-offset',type=float, help='Voltage around where the oscillation takes place.',default=3.75)
parser.add_argument('-nocalib', help ='Allows calibration of offset voltage', action = 'store_false', default = True)
parser.add_argument('-compar', help ='Gives a visual of the exit voltage', action = 'store_true', default = False)
args = parser.parse_args()

#Paramters for acquisition (shortcut variables)
fqreal = int(args.fq/float(args.wvfrmres))
scanres = args.readsize * 127
scanrate = fqreal*127


if int(args.fq/float(args.wvfrmres)) != args.fq/float(args.wvfrmres):
	raise AssertionError('Waveform resolution not multiple of frequency')
if args.readnum <= 1:
	raise AssertionError('Must read stream more than once for stability')
if args.compar == True and args.readnum>=10000:
	raise AssertionError('Cannot give visual for very large loop')


#Waveform generation
def initialise_sinus(periods,amplitude,offset):
	sp = np.linspace(0, int(periods) * 2 * np.pi,128)
	val = (amplitude * np.sin(sp))+offset
	return val

def initialise_streamout(handle,waveform,size=1):
	'''Intialise a labjack handle object open for Stream-Out with a buffer sufficient for a loop size of 127 values on DAC0 from a waveform iterable given. If the loop is bigger than 127, specify the multiplier with the size variable'''

	#Writing constants for Stream-Out
	ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",0)
	ljm.eWriteName(handle,"STREAM_OUT0_TARGET",1000)
	ljm.eWriteName(handle,"STREAM_OUT0_BUFFER_SIZE",256*size)
	ljm.eWriteName(handle,"STREAM_OUT0_ENABLE",1)

	#Writing the waveform onto the labjack
	ljm.eWriteName(handle,"STREAM_OUT0_LOOP_SIZE",size*127)
	for x in range(size):
		ljm.eWriteNameArray(handle,"STREAM_OUT0_BUFFER_F32",127,waveform[x*128:(x+1)*127])
	ljm.eWriteName(handle,"STREAM_OUT0_SET_LOOP",1)


def mainloop_reading(handle,scanres,readnum,readsize,wvfrmres,fq,verbosity,compar):
	'''
	Mainloop used for reading values and calculating the distance between two peaks. Could yield the values as they are calculating.

	handle variable is the labjack object
	scanres normaly equals to scanres from eStreamStart method
	channels gives out the number of channels that need to be read by the function
	readnum is the the number of time the while function will be read.
	'''
	count = 0
	if compar == True:
		data = np.empty(scanres*(readnum-1))
	while True:
		read = ljm.eStreamRead(handle)
		if compar == True:
			if count > 0:data[(count-1)*(scanres):count*scanres] = read[0][0:scanres]

		#Peak detection
		if count>0:
		 	dataaverage =  np.array([np.mean([(read[0][0:scanres])[y + (x*127/wvfrmres)] for x in range((readsize*wvfrmres) -1)]) for y in range(127/wvfrmres)]) 
		 	peakind = peak_detect_periodic(dataaverage,fq,scanrate)
			if len(peakind[0])==2:
				print('Ratio: {0}'.format((peakind[0][1]-peakind[0][0])*(wvfrmres*2)/127.0))
			else:
				if args.verbosity >0:print('NaN because of {0} spike(s) detected'.format(len(peakind[0])))
		if verbosity > 0:
			print('DeviceBufferStatus: {0}  LJMBufferStatus:{1}'.format(read[1],read[2]))
		count += 1
		if count >= readnum:
			if verbosity > 0 : print("Number of read obtained:{0}".format(count))
			break
		time.sleep(1/float(scanres * scanrate/127))
	if compar == True:
		return data

if __name__ == "__main__":
	#Opening labjack and generating handle object
	try:
		handle = ljm.openS("T7","ETHERNET","470010370")
	except ljm.LJMError: 
		print('Could not open labjack \n Program is terminated') 
		sys.exit()
	try:
		if args.nocalib == False:
			val = initialise_sinus(args.wvfrmres,args.amp, args.offset)
		else:
			val = initialise_sinus(args.wvfrmres,args.amp,offset_calibration(handle))

		initialise_streamout(handle,val)

		#Start streaming. 4800 is Stream-Out. 0 is Stream-In
		if args.verbosity > 0: print("Streaming initiated")
		
		ljm.eStreamStart(handle,scanres,2,[4800,0],scanrate)

		data = mainloop_reading(handle,scanres,args.readnum,args.readsize,args.wvfrmres,args.fq,args.verbosity,args.compar)

	#Error catching
	except ljm.LJMError:
		print("LJM Error break")
	except Exception:
		print("System error break")
		print(traceback.format_exc())
		sys.exit()
	except KeyboardInterrupt:
		if args.verbosity > 0 : print("\n User called break")
	try:
		ljm.eStreamStop(handle)
		if args.verbosity > 0: print("Streaming finished")
	except ljm.LJMError: 
		print('Could not stop stream')
	try:
		ljm.close(handle)
		if args.verbosity >0: print("Labjack closed")
	except ljm.LJMError: 
		print('Could not close labjack')
		sys.exit()
	

	if args.compar == True:
		#Initialising x-axis for plotting
		plotspace = np.linspace(0,((args.readnum-1)*args.readsize)/(float(fqreal)),scanres*(args.readnum-1))
		plotspace2 = np.linspace(0,(1)/(float(args.fq)),127/args.wvfrmres)

		#Converting theoritical sine wave to square wave

		theofunction = np.sin(np.linspace(0,args.wvfrmres*(2*np.pi)*(args.readnum-1)*args.readsize, scanres*(args.readnum-1)))
		square = [1 if x>=np.mean(theofunction) else 0 for x in theofunction]

		#Visualisation of the peaks and function created
		spectrum = np.fft.rfft(data)
		freqseq = np.fft.rfftfreq(len(spectrum),1/float(scanrate))
		spectrum = spectrum * np.exp(-np.power(freqseq - (args.fq+2*args.fq)/2 , 2.) / (2 * np.power(400, 2.)))
		spectrum[np.where(np.logical_or(np.less(abs(freqseq),args.fq*0.2),np.greater(abs(freqseq),args.fq*13)))] = 0
		datamod = np.real(np.fft.irfft(spectrum))
		peak = np.zeros(len(data))
		peak[peak_detect_periodic(data,args.fq,scanrate,sensibility=0)] = 5

		#mpl.plot(plotspace, square)
		mpl.step(plotspace, normalize_set(data,square))
		mpl.plot(plotspace, normalize_set(datamod,square))
		mpl.plot(plotspace, normalize_set(peak,square))
		mpl.xlim(0,0.05)
		mpl.show()
