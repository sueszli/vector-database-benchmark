import numpy as np
import matplotlib.pyplot as plt

filename1 = "ExperimentOptotuneDiffuser.txt"
filename2 = "diffuserTransmissionTheoretical.txt"
filename3 = "DG10-600_powermeter.txt"
filename4 = "DG10-600_powermeter2.txt"
filename5 = "DG10-120_powermeter.txt"

distanceDiffuserScreen1 = 36  # mm
distanceDiffuserScreen2 = 46  # mm
distanceDiffuserScreen3 = 50  # mm
distanceDiffuserScreen4 = 60  # mm
distanceDiffuserScreen5 = 70  # mm

# distances for DG10-120
distanceDiffuserScreen6 = 75
distanceDiffuserScreen7 = 85
distanceDiffuserScreen8 = 95


# Load les files, transformer une colonne quelconque en liste que tu peux utiliser sur python
# x : distances
# y : puissances
# a : angles
x1 = np.loadtxt(filename1, usecols=0)
y1 = np.loadtxt(filename1, usecols=1)
y2 = np.loadtxt(filename1, usecols=2)
y3 = np.loadtxt(filename1, usecols=3)
y4 = np.loadtxt(filename1, usecols=4)
y5 = np.loadtxt(filename1, usecols=5)
a6 = np.loadtxt(filename2, usecols=0)
y6 = np.loadtxt(filename2, usecols=1)
x2 = np.loadtxt(filename4, usecols=0)
y7 = np.loadtxt(filename3, usecols=1)
y8 = np.loadtxt(filename4, usecols=1)

x3 = np.loadtxt(filename5, usecols=0)
y9 = np.loadtxt(filename5, usecols=1)
y10 = np.loadtxt(filename5, usecols=2)
y11 = np.loadtxt(filename5, usecols=3)


# Transformer les distances en angle avec la puissance mesurée avec le powermeter.
# degrees : Fonction qui transforme des radians en degrées.
# SOCAHTOAH
# yn : y normalisé
a1 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen1)) + 4
y1n = np.array(y1)/max(y1)
a2 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen2)) + 4
y2n = np.array(y2)/max(y2)
a3 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen3)) + 2.22
y3n = np.array(y3)/max(y3)
a4 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen4)) + 2.22
y4n = np.array(y4)/max(y4)
a5 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen5)) + 2.22
y5n = np.array(y5)/max(y5)
a7 = np.degrees(np.arctan((np.array(x1) - x1[-1]/2)/distanceDiffuserScreen5)) + 5.2
a8 = np.degrees(np.arctan((np.array(x2) - x2[-1]/2)/distanceDiffuserScreen3)) + 6.8
y7n = np.array(y7)/max(y7)
y8n = np.array(y8)/max(y8)

a9 = np.degrees(np.arctan((np.array(x3) - x3[-1]/2)/distanceDiffuserScreen6)) - 2.5
a10 = np.degrees(np.arctan((np.array(x3) - x3[-1]/2)/distanceDiffuserScreen7)) - 2.5
a11 = np.degrees(np.arctan((np.array(x3) - x3[-1]/2)/distanceDiffuserScreen8)) - 2.5
y9n = np.array(y9)/max(y9)
y10n = np.array(y10)/max(y10)
y11n = np.array(y11)/max(y11)

# Affecter les courbes à une variable, à quoi tu veux que ça ressemble.
Data0, = plt.plot(a1, y1n, color='k', label='Grit 1500 3.6 cm')
Data1, = plt.plot(a2, y2n, color='k', label='Grit 1500 4.6 cm', linestyle='--')
Data2, = plt.plot(a3, y3n, color='b', label='Optotune Diffuser 5 cm')
Data3, = plt.plot(a4, y4n, color='b', label='Optotune Diffuser 6 cm', linestyle = '--')
Data4, = plt.plot(a5, y5n, color='b', label='Optotune Diffuser 7 cm', linestyle = ':')
Data5, = plt.plot(a6, y6, color='r', label='Thorlabs Data', linestyle=':')
Data6, = plt.plot(a7, y7n, label='Grit 600 7 cm')
Data7, = plt.plot(a8, y8n, label='Grit 600 5 cm')

Data8, = plt.plot(a9, y9n, color='g', label = 'Grit 120 7.5 cm')
Data9, = plt.plot(a10, y10n, color = 'g', label = 'Grit 120 8.5 cm', linestyle = '--')
Data10, = plt.plot(a11, y11n, color = 'g', label = 'Grit 120 9.5 cm', linestyle = ':')

# Le graphe en lui-même ressemble à quoi.
plt.xlim([a3[0], a3[-1]])
plt.ylim([0, 1.05])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Output Angle [°]", fontsize=18)
plt.ylabel("Normalized power [-]", fontsize=18)
plt.grid(False)
plt.legend(handles=[Data0, Data1, Data2, Data3, Data4, Data5, Data6, Data7, Data8, Data9, Data10], fontsize=16)
plt.minorticks_on()
fig = plt.gcf()
fig.set_size_inches(12, 7)

# Voir ou save ; un des deux.
# fig.savefig('graphDiffuser1500gritPowermeter.pdf', bbox_inches='tight', dpi=600)
plt.show()