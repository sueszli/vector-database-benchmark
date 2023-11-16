"""Modified from:

https://raw.githubusercontent.com/jradavenport/cubehelix/master/cubehelix.py

Copyright (c) 2014, James R. A. Davenport and contributors All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from math import pi
import numpy as np

def cubehelix(start=0.5, rot=1, gamma=1.0, reverse=True, nlev=256.0, minSat=1.2, maxSat=1.2, minLight=0.0, maxLight=1.0, **kwargs):
    if False:
        while True:
            i = 10
    '\n    A full implementation of Dave Green\'s "cubehelix" for Matplotlib.\n    Based on the FORTRAN 77 code provided in\n    D.A. Green, 2011, BASI, 39, 289.\n\n    http://adsabs.harvard.edu/abs/2011arXiv1108.5083G\n\n    User can adjust all parameters of the cubehelix algorithm.\n    This enables much greater flexibility in choosing color maps, while\n    always ensuring the color map scales in intensity from black\n    to white. A few simple examples:\n\n    Default color map settings produce the standard "cubehelix".\n\n    Create color map in only blues by setting rot=0 and start=0.\n\n    Create reverse (white to black) backwards through the rainbow once\n    by setting rot=1 and reverse=True.\n\n    Parameters\n    ----------\n    start : scalar, optional\n        Sets the starting position in the color space. 0=blue, 1=red,\n        2=green. Defaults to 0.5.\n    rot : scalar, optional\n        The number of rotations through the rainbow. Can be positive\n        or negative, indicating direction of rainbow. Negative values\n        correspond to Blue->Red direction. Defaults to -1.5\n    gamma : scalar, optional\n        The gamma correction for intensity. Defaults to 1.0\n    reverse : boolean, optional\n        Set to True to reverse the color map. Will go from black to\n        white. Good for density plots where shade~density. Defaults to False\n    nlev : scalar, optional\n        Defines the number of discrete levels to render colors at.\n        Defaults to 256.\n    sat : scalar, optional\n        The saturation intensity factor. Defaults to 1.2\n        NOTE: this was formerly known as "hue" parameter\n    minSat : scalar, optional\n        Sets the minimum-level saturation. Defaults to 1.2\n    maxSat : scalar, optional\n        Sets the maximum-level saturation. Defaults to 1.2\n    startHue : scalar, optional\n        Sets the starting color, ranging from [0, 360], as in\n        D3 version by @mbostock\n        NOTE: overrides values in start parameter\n    endHue : scalar, optional\n        Sets the ending color, ranging from [0, 360], as in\n        D3 version by @mbostock\n        NOTE: overrides values in rot parameter\n    minLight : scalar, optional\n        Sets the minimum lightness value. Defaults to 0.\n    maxLight : scalar, optional\n        Sets the maximum lightness value. Defaults to 1.\n\n    Returns\n    -------\n    data : ndarray, shape (N, 3)\n        Control points.\n    '
    if kwargs is not None:
        if 'startHue' in kwargs:
            start = (kwargs.get('startHue') / 360.0 - 1.0) * 3.0
        if 'endHue' in kwargs:
            rot = kwargs.get('endHue') / 360.0 - start / 3.0 - 1.0
        if 'sat' in kwargs:
            minSat = kwargs.get('sat')
            maxSat = kwargs.get('sat')
    fract = np.linspace(minLight, maxLight, nlev)
    angle = 2.0 * pi * (start / 3.0 + rot * fract + 1.0)
    fract = fract ** gamma
    satar = np.linspace(minSat, maxSat, nlev)
    amp = satar * fract * (1.0 - fract) / 2.0
    red = fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
    grn = fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
    blu = fract + amp * (1.97294 * np.cos(angle))
    red[np.where(red > 1.0)] = 1.0
    grn[np.where(grn > 1.0)] = 1.0
    blu[np.where(blu > 1.0)] = 1.0
    red[np.where(red < 0.0)] = 0.0
    grn[np.where(grn < 0.0)] = 0.0
    blu[np.where(blu < 0.0)] = 0.0
    if reverse is True:
        red = red[::-1]
        blu = blu[::-1]
        grn = grn[::-1]
    return np.array((red, grn, blu)).T