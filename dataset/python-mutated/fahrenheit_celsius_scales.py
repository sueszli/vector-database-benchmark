"""
=================================
Different scales on the same axes
=================================

Demo of how to display two scales on the left and right y-axis.

This example uses the Fahrenheit and Celsius scales.
"""
import matplotlib.pyplot as plt
import numpy as np

def fahrenheit2celsius(temp):
    if False:
        return 10
    '\n    Returns temperature in Celsius given Fahrenheit temperature.\n    '
    return 5.0 / 9.0 * (temp - 32)

def make_plot():
    if False:
        while True:
            i = 10

    def convert_ax_c_to_celsius(ax_f):
        if False:
            while True:
                i = 10
        '\n        Update second axis according to first axis.\n        '
        (y1, y2) = ax_f.get_ylim()
        ax_c.set_ylim(fahrenheit2celsius(y1), fahrenheit2celsius(y2))
        ax_c.figure.canvas.draw()
    (fig, ax_f) = plt.subplots()
    ax_c = ax_f.twinx()
    ax_f.callbacks.connect('ylim_changed', convert_ax_c_to_celsius)
    ax_f.plot(np.linspace(-40, 120, 100))
    ax_f.set_xlim(0, 100)
    ax_f.set_title('Two scales: Fahrenheit and Celsius')
    ax_f.set_ylabel('Fahrenheit')
    ax_c.set_ylabel('Celsius')
    plt.show()
make_plot()