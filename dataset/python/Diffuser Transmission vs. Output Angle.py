import numpy as np
import matplotlib.pyplot as plt

filename1 = "diffuserTransmissionAngleData0.1mW.txt"
filename2 = "diffuserTransmissionAngleData1mW.txt"
filename3 = "diffuserTransmissionTheoretical.txt"
distanceDiffuserScreen = 60  # mm

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)

x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)

a1 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen))
y1n = np.array(y1)/max(y1)

a2 = np.degrees(np.arctan((np.array(x2) - x2[-1]/2)/distanceDiffuserScreen))
y2n = np.array(y2)/max(y2)

x3 = np.loadtxt(filename3, usecols=0)
y3 = np.loadtxt(filename3, usecols=1)

Data0, = plt.plot(a1, y1n, color='k', label='Laser power = 0.1 mW')
Data1, = plt.plot(a2, y2n, color='k', label='Laser power = 1 mW', linestyle='--')
Data2, = plt.plot(x3, y3, color='b', label='Theoretical transmission')
plt.xlim([a1[0], a1[-1]])
plt.ylim([0, 1.05])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Output Angle [°]", fontsize=18)
plt.ylabel("Normalized gray value [-]", fontsize=18)
plt.grid(True)
plt.legend(handles=[Data0, Data1, Data2], fontsize=16)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.savefig('graphdiffuser.pdf', bbox_inches='tight', dpi=600)
#plt.show()