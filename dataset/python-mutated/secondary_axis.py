"""
==============
Secondary Axis
==============

Sometimes we want a secondary axis on a plot, for instance to convert
radians to degrees on the same plot.  We can do this by making a child
axes with only one axis visible via `.axes.Axes.secondary_xaxis` and
`.axes.Axes.secondary_yaxis`.  This secondary axis can have a different scale
than the main axis by providing both a forward and an inverse conversion
function in a tuple to the *functions* keyword argument:
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
(fig, ax) = plt.subplots(layout='constrained')
x = np.arange(0, 360, 1)
y = np.sin(2 * x * np.pi / 180)
ax.plot(x, y)
ax.set_xlabel('angle [degrees]')
ax.set_ylabel('signal')
ax.set_title('Sine wave')

def deg2rad(x):
    if False:
        print('Hello World!')
    return x * np.pi / 180

def rad2deg(x):
    if False:
        print('Hello World!')
    return x * 180 / np.pi
secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
secax.set_xlabel('angle [rad]')
plt.show()
(fig, ax) = plt.subplots(layout='constrained')
x = np.arange(0.02, 1, 0.02)
np.random.seed(19680801)
y = np.random.randn(len(x)) ** 2
ax.loglog(x, y)
ax.set_xlabel('f [Hz]')
ax.set_ylabel('PSD')
ax.set_title('Random spectrum')

def one_over(x):
    if False:
        return 10
    'Vectorized 1/x, treating x==0 manually'
    x = np.array(x, float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x
inverse = one_over
secax = ax.secondary_xaxis('top', functions=(one_over, inverse))
secax.set_xlabel('period [s]')
plt.show()
(fig, ax) = plt.subplots(layout='constrained')
xdata = np.arange(1, 11, 0.4)
ydata = np.random.randn(len(xdata))
ax.plot(xdata, ydata, label='Plotted data')
xold = np.arange(0, 11, 0.2)
xnew = np.sort(10 * np.exp(-xold / 4) + np.random.randn(len(xold)) / 3)
ax.plot(xold[3:], xnew[3:], label='Transform data')
ax.set_xlabel('X [m]')
ax.legend()

def forward(x):
    if False:
        for i in range(10):
            print('nop')
    return np.interp(x, xold, xnew)

def inverse(x):
    if False:
        for i in range(10):
            print('nop')
    return np.interp(x, xnew, xold)
secax = ax.secondary_xaxis('top', functions=(forward, inverse))
secax.xaxis.set_minor_locator(AutoMinorLocator())
secax.set_xlabel('$X_{other}$')
plt.show()
dates = [datetime.datetime(2018, 1, 1) + datetime.timedelta(hours=k * 6) for k in range(240)]
temperature = np.random.randn(len(dates)) * 4 + 6.7
(fig, ax) = plt.subplots(layout='constrained')
ax.plot(dates, temperature)
ax.set_ylabel('$T\\ [^oC]$')
plt.xticks(rotation=70)

def date2yday(x):
    if False:
        i = 10
        return i + 15
    'Convert matplotlib datenum to days since 2018-01-01.'
    y = x - mdates.date2num(datetime.datetime(2018, 1, 1))
    return y

def yday2date(x):
    if False:
        for i in range(10):
            print('nop')
    'Return a matplotlib datenum for *x* days after 2018-01-01.'
    y = x + mdates.date2num(datetime.datetime(2018, 1, 1))
    return y
secax_x = ax.secondary_xaxis('top', functions=(date2yday, yday2date))
secax_x.set_xlabel('yday [2018]')

def celsius_to_fahrenheit(x):
    if False:
        print('Hello World!')
    return x * 1.8 + 32

def fahrenheit_to_celsius(x):
    if False:
        while True:
            i = 10
    return (x - 32) / 1.8
secax_y = ax.secondary_yaxis('right', functions=(celsius_to_fahrenheit, fahrenheit_to_celsius))
secax_y.set_ylabel('$T\\ [^oF]$')

def celsius_to_anomaly(x):
    if False:
        while True:
            i = 10
    return x - np.mean(temperature)

def anomaly_to_celsius(x):
    if False:
        for i in range(10):
            print('nop')
    return x + np.mean(temperature)
secax_y2 = ax.secondary_yaxis(1.2, functions=(celsius_to_anomaly, anomaly_to_celsius))
secax_y2.set_ylabel('$T - \\overline{T}\\ [^oC]$')
plt.show()