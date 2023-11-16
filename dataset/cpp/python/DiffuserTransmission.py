import numpy as np
import matplotlib.pyplot as plt

filename1 = "DiffuserTransmission\\Transmission_17_diffuser.txt"
filename2 = "DiffuserTheoreticalTransmission\\DiffuserTransmission.txt"

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1) #+ 53.77

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

Data0, = plt.plot(x1, y1, color='k', label='Experimental')
Data1, = plt.plot(x2, y2, color='b', label='Theoretical')
plt.xlim([450, 650])
plt.ylim([0, 100])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Wavelength [nm]", fontsize=18)
plt.ylabel("Transmission [%]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1], fontsize=18)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
#fig.savefig('DiffuserTransmissionRaw.pdf', bbox_inches='tight', dpi=600)
plt.show()