import numpy as np 
import scipy.signal 

def normalize_set(set1,set2):
	'''Normalize set1 on set 2'''
	max1 = np.amax(set1)
	min1 = np.amin(set1)
	diff1 =max1-min1
	max2 = np.amax(set2)
	min2 = np.amin(set2)
	diff2 = max2-min2
	return (((diff2)/float(diff1))*(set1-(((max1-min1)/2) + min1)))+(((max2-min2)/2) + min2)

def shift_set(set1,set2,rate,normalize=False,giveshift=False):
	'''Shifts set1 to the phase of set2. set1 and set2 must have an even number of elements'''
	rfft1 = np.fft.rfft(set1)
	rfft2 = np.fft.rfft(set2)
	max1 = np.argmax(abs(rfft1[1:]))
	max2 = np.argmax(abs(rfft2[1:]))
	shiftangle = (np.angle(rfft2[1:][max2]) - np.angle(rfft1[1:][max1]))
	max1 = max1+1
	rfft1[max1] = rfft1[max1] * np.exp(shiftangle* 1j) * abs(rfft1[max1])/float(abs(rfft1[max1] * np.exp(shiftangle *1j)))
	set1shift = abs(np.fft.irfft(rfft1))
	if normalize == True or giveshift == True:
		if normalize == True:
			set1shift = normalize_set(set1shift,set2)
		if giveshift == True:
			return (set1shift, shiftangle*1/float(rate))
	return set1shift

def find_dominantfq(freqseq,spectrum,rate):
	mask = abs(spectrum[1:]) > 0.5*max(abs(spectrum[1:]))
	peaks = freqseq[mask]
	return peaks