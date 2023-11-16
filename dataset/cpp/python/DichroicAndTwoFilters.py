import numpy as np
import matplotlib.pyplot as plt

filename1 = "Filter_500-590\\Transmission_5_500590.txt"
filename2 = "Filter_D480-500\\Transmission_9_480500.txt"
filename3 = "FF500-Di01_Semrock\\Transmission_13_emission.txt"

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

x3 = np.loadtxt(filename3, usecols=0)
y3 = np.loadtxt(filename3, usecols=1)

Data0, = plt.plot(x1, y1, color='g', label='Emission filter 500-590')
Data1, = plt.plot(x2, y2, color='b', label='Excitation filter 480-500')
Data2, = plt.plot(x3, y3, color='k', label='Dichroic mirror')
plt.xlim([470, 600])
plt.ylim([0, 100])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Wavelength [nm]", fontsize=18)
plt.ylabel("Transmission [%]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2], fontsize=18)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
#fig.savefig('DichroicAndTwoFilters.pdf', bbox_inches='tight', dpi=600)
plt.show()