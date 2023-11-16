"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np

def mag_to_M0(mag):
    if False:
        while True:
            i = 10
    M0 = 10.0 ** (1.5 * mag + 9.1)
    return M0