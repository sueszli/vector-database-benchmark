# coding=utf-8
import numpy as np
from scipy import signal

def normalize_set(set1,set2):
	'''Normalize set1 on set 2'''
	max1 = np.amax(set1)	# retourne la valeur maximale du set1
	min1 = np.amin(set1)	# retourne la valeur minimale du set1
	diff1 = max1-min1
	max2 = np.amax(set2)	# retourne la valeur maximale du set2
	min2 = np.amin(set2)	# retourne la valeur minimale du set2
	diff2 = max2-min2
	return (((diff2)/float(diff1))*(set1-(((diff1)/2) + min1)))+(((diff2)/2) + min2)

def peak_detect_periodic(set,fq,scanrate,sensibility=0.1):
	'''Returns all the peak detection for a periodic signal of a given frequency. May not work.'''
	spectrum = np.fft.fft(set)										# Faire deux fois la Transformée de Fourier discrète?
	freqseq = np.fft.fftfreq(len(spectrum),1/float(scanrate))		# retourne un tableau de fréquences des éléments de sortie correspondants
	spectrum = spectrum * np.exp(-np.power(freqseq - (fq+2*fq)/2 , 2.) / (2 * np.power(300, 2.)))
	spectrum[np.where(np.logical_or(np.less(abs(freqseq),fq*0.2),np.greater(abs(freqseq),fq*13)))] = 0
	setmod = np.real(np.fft.ifft(spectrum))		# Retourne la partie réelle de l'argument. Si l'argument est réel, retourne le type de l'argument. Sinon, (si l'argument est complexe), le type de l'argument est float.
	peakind = np.array(signal.argrelmax(np.array([x if x>sensibility*setmod.max() else 0 for x in setmod])))
	return peakind

if __name__ == "__main__":
	pass