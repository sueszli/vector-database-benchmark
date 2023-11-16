import numpy as np
import matplotlib.pyplot as plt

filename1 = "DG1500.txt"
filename2 = "DG600.txt"
filename3 = "DG220.txt"
filename4 = "DG120.txt"
filename5 = "DG1500at85mm.txt"

distanceDiffuserScreen1 = 60  # mm
distanceDiffuserScreen2 = 85  # mm

x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)
x2 = np.loadtxt(filename2, usecols=0)
y2 = np.loadtxt(filename2, usecols=1)
x3 = np.loadtxt(filename3, usecols=0)
y3 = np.loadtxt(filename3, usecols=1)
x4 = np.loadtxt(filename4, usecols=0)
y4 = np.loadtxt(filename4, usecols=1)
x5 = np.loadtxt(filename5, usecols=0)
y5 = np.loadtxt(filename5, usecols=1)

a1 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen1))
y1n = np.array(y1)/max(y1)
a2 = np.degrees(np.arctan((np.array(x2) - x2[-1]/2)/distanceDiffuserScreen1))
y2n = np.array(y2)/max(y2)
a3 = np.degrees(np.arctan((np.array(x3) - x3[-1]/2)/distanceDiffuserScreen1))
y3n = np.array(y3)/max(y3)
a4 = np.degrees(np.arctan((np.array(x4) - x4[-1]/2)/distanceDiffuserScreen1))
y4n = np.array(y4)/max(y4)
a5 = np.degrees(np.arctan((np.array(x5) - x5[-1]/2)/distanceDiffuserScreen2))
y5n = np.array(y5)/max(y5)

Data0, = plt.plot(a1, y1n, color='k', label='Grit 1500')
Data1, = plt.plot(a2, y2n, color='k', label='Grit 600', linestyle='--')
Data2, = plt.plot(a3, y3n, color='b', label='Grit 220')
Data3, = plt.plot(a4, y4n, color='g', label='Grit 120')
Data4, = plt.plot(a5, y5n, color='r', label='Grit 1500 (85 mm)')
plt.xlim([a1[0], a1[-1]])
plt.ylim([0, 1.05])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Output Angle [°]", fontsize=18)
plt.ylabel("Normalized gray value (Laser power = 1 mW) [-]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2, Data3, Data4], fontsize=16)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.savefig('graphdiffuser.pdf', bbox_inches='tight', dpi=600)
plt.show()