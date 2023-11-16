import numpy as np
import matplotlib.pyplot as plt

filename = "DiffusersData.txt"

x1 = np.loadtxt(filename, usecols=0, skiprows=1)
y1 = np.loadtxt(filename, usecols=1, skiprows=1) #- 10472.871
x2 = np.loadtxt(filename, usecols=2, skiprows=1)
y2 = np.loadtxt(filename, usecols=3, skiprows=1) #- 12223.191
x3 = np.loadtxt(filename, usecols=4, skiprows=1)
y3 = np.loadtxt(filename, usecols=5, skiprows=1) #- 12854.000
x4 = np.loadtxt(filename, usecols=6, skiprows=1)
y4 = np.loadtxt(filename, usecols=7, skiprows=1) #- 14197.538
x5 = np.loadtxt(filename, usecols=8, skiprows=1)
y5 = np.loadtxt(filename, usecols=9, skiprows=1) #- 16592.633

Data0, = plt.plot(x1, y1, color='k', label='120 grit')
Data1, = plt.plot(x2, y2, color='g', label='220 grit-polished side output')
Data2, = plt.plot(x3, y3, color='b', label='220 grit-polished side input')
Data3, = plt.plot(x4, y4, color='y', label='600 grit')
Data4, = plt.plot(x5, y5, color='r', label='1500 grit')
plt.xlim([x1[0], x1[-1]])
#plt.ylim([0, 1.05])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Pixels [-]", fontsize=18)
plt.ylabel("Gray value [-]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2, Data3, Data4], fontsize=16)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)
fig.savefig('graphdiffuser.pdf', bbox_inches='tight', dpi=600)
#plt.show()