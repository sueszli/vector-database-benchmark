import numpy as np
import matplotlib.pyplot as plt

filename1 = "Cobolt_laser\\FLMS002391_11-31-24-256 laser.txt"
filename2 = "Filter_D480-500\\Transmission_9_480500.txt"
filename3 = "GFP_spectrum\\GFP_spectra.txt"

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)
y1n = np.array(y1)/max(y1)*100

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

x3 = np.genfromtxt(fname=filename3, usecols=(0,), delimiter=',')
y3 = np.genfromtxt(fname=filename3, usecols=(1,), delimiter=',')*100

Data0, = plt.plot(x1, y1n, color='k', label='Laser spectrum')
Data1, = plt.plot(x2, y2, color='k', label='Excitation filter 480-500', linestyle='--')
Data2, = plt.plot(x3, y3, color='b', label='EGFP excitation')
plt.xlim([440, 540])
plt.ylim([0, 101])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Wavelength [nm]", fontsize=18)
plt.ylabel("Transmission [%]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2], fontsize=18)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
#fig.savefig('LaserExcitation.pdf', bbox_inches='tight', dpi=600)
plt.show()