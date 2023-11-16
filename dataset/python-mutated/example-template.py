"""
========================
Title of Example
========================

This example <verb> <active tense> <does something>.

The example uses <packages> to <do something> and <other package> to <do other
thing>. Include links to referenced packages like this: `astropy.io.fits` to
show the astropy.io.fits or like this `~astropy.io.fits`to show just 'fits'


*By: <names>*

*License: BSD*


"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
x = np.linspace(-np.pi, np.pi, 300)
(xx, yy) = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)
plt.figure()
plt.imshow(z)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('hot'))
plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('Spectral'), interpolation='none')

def dummy() -> None:
    if False:
        while True:
            i = 10
    "Dummy function to make sure docstrings don't get rendered as text"
string = "\nTriple-quoted string which tries to break parser but doesn't.\n"
print('Some output from Python')
plt.show()