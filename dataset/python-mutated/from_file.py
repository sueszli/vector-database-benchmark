import sys
import numpy as np
from astropy import wcs
from astropy.io import fits

def load_wcs_from_file(filename):
    if False:
        while True:
            i = 10
    hdulist = fits.open(filename)
    w = wcs.WCS(hdulist[0].header)
    print(w.wcs.name)
    w.wcs.print_contents()
    pixcrd = np.array([[0, 0], [24, 38], [45, 98]], dtype=np.float64)
    world = w.wcs_pix2world(pixcrd, 0)
    print(world)
    pixcrd2 = w.wcs_world2pix(world, 0)
    print(pixcrd2)
    assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-06
    x = 0
    y = 0
    origin = 0
    assert w.wcs_pix2world(x, y, origin) == w.wcs_pix2world(x + 1, y + 1, origin + 1)
if __name__ == '__main__':
    load_wcs_from_file(sys.argv[-1])