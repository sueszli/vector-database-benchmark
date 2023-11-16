from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS

def download_image_save_cutout(url, position, size):
    if False:
        return 10
    filename = download_file(url)
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header)
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    hdu.data = cutout.data
    hdu.header.update(cutout.wcs.to_header())
    cutout_filename = 'example_cutout.fits'
    hdu.writeto(cutout_filename, overwrite=True)
if __name__ == '__main__':
    url = 'https://astropy.stsci.edu/data/photometry/spitzer_example_image.fits'
    position = (500, 300)
    size = (400, 400)
    download_image_save_cutout(url, position, size)