"""
=============
Custom Ticker
=============

The :mod:`matplotlib.ticker` module defines many preset tickers, but was
primarily designed for extensibility, i.e., to support user customized ticking.

In this example, a user defined function is used to format the ticks in
millions of dollars on the y-axis.
"""
import matplotlib.pyplot as plt

def millions(x, pos):
    if False:
        print('Hello World!')
    'The two arguments are the value and tick position.'
    return f'${x * 1e-06:1.1f}M'
(fig, ax) = plt.subplots()
ax.yaxis.set_major_formatter(millions)
money = [150000.0, 2500000.0, 5500000.0, 20000000.0]
ax.bar(['Bill', 'Fred', 'Mary', 'Sue'], money)
plt.show()