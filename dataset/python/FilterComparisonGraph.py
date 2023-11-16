import numpy as np
import matplotlib.pyplot as plt

filename1 = "Filter_BA510-550\\Transmission_1_510550.txt"
filename2 = "Filter_500-590\\Transmission_5_500590.txt"
filename3 = "GFP_spectrum\\GFP_spectra.txt"

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

x3 = np.genfromtxt(fname=filename3, usecols=(0,), delimiter=',')
y3 = np.genfromtxt(fname=filename3, usecols=(2,), delimiter=',')*100

Data0, = plt.plot(x3, y3, color='g', label='EGFP emission')
Data1, = plt.plot(x1, y1, color='k', label='Emission filter 510-550')
Data2, = plt.plot(x2, y2, color='k', label='Emission filter 500-590', linestyle='--')
plt.xlim([450, 650])
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
#fig.savefig('FilterComparison.pdf', bbox_inches='tight', dpi=600)
plt.show()