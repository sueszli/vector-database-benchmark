from labjack import ljm
import numpy as np
import time
import matplotlib.pyplot as mpl
from scipy import signal
import traceback
import sys
import argparse

#External function files
from labjackcontrol import *
from signal_processing_functions import *


def offset_calibration(handle,precision=2):
	'''With a linear sweep of every possible tension, the program finds the maximum voltage read'''
	linear = np.arange(2.5,5,2.5/((4*127)+1))
	initialise_streamout(handle,linear,4)
	ljm.eStreamStart(handle,(4*127),2,[4800,0],150)
	data = np.empty(4*127)
	for x in range(precision):
		read = ljm.eStreamRead(handle)
		data = np.append(data,read[0][0:4*127])
	ljm.eStreamStop(handle)
	slope = ((2.5 * np.linspace(4*127,2*4*127,4*127))/(4*127))
	data = np.convolve(data,signal.gaussian(4*127,10),'same')[(4*127)-1:(2*4*127)-1]
	peakind = np.array(signal.argrelmax(np.array([x if x>0.99*data.max() else 0 for x in data])))
	if peakind.mean()<len(slope):
		maxVoltage = slope[peakind.mean()]
		print('Maximum found for {0}'.format(maxVoltage))
	else:
		print('No maxima found, 3.75 will be the offset')
		maxVoltage = 3.75
	return maxVoltage