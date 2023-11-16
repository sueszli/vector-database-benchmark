import numpy as np
import matplotlib.pyplot as plt

filename1 = "EmissionFilter_520-28\\520-28_1.txt"
filename2 = "ExcitationFilter_482-18\\482-18_1.txt"
filename3 = "Dichroic_FF495-Di03-25x36\\FF495-Di03-25x36 (1).txt"
filename4 = "EmissionFilter_520-28\\Theoretical_EmissionFilter.txt"
filename5 = "ExcitationFilter_482-18\\Theoretical_ExcitationFilter.txt"
filename6 = "Dichroic_FF495-Di03-25x36\\Theoretical_Dichroic.txt"

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

x3 = np.loadtxt(filename3, usecols=0)
y3 = np.loadtxt(filename3, usecols=1)

x4 = np.loadtxt(filename4, usecols=0) #- 8  #because the spectrometer is crap
y4 = np.loadtxt(filename4, usecols=1)*100

x5 = np.loadtxt(filename5, usecols=0) #- 8
y5 = np.loadtxt(filename5, usecols=1)*100

x6 = np.loadtxt(filename6, usecols=0) #- 8
y6 = np.loadtxt(filename6, usecols=1)*100

Data0, = plt.plot(x1, y1, color='g', label='Emission filter 520-28')
Data1, = plt.plot(x2, y2, color='b', label='Excitation filter 482-18')
Data2, = plt.plot(x3, y3, color='k', label='Dichroic mirror')
Data3, = plt.plot(x4, y4, color='g', label='Theoretical Emission filter', linestyle='--')
Data4, = plt.plot(x5, y5, color='b', label='Theoretical Excitation filter', linestyle='--')
Data5, = plt.plot(x6, y6, color='k', label='Theoretical Dichroic', linestyle='--')
plt.xlim([450, 600])
plt.ylim([0, 100])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Wavelength [nm]", fontsize=18)
plt.ylabel("Transmission [%]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2, Data3, Data4, Data5], fontsize=18)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.savefig('DichroicAndTwoFiltersPackage.pdf', bbox_inches='tight', dpi=600)
#plt.show()