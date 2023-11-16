import numpy as np
from astropy.io import fits
from .conftest import FitsTestCase

class TestNonstandardHdus(FitsTestCase):

    def test_create_fitshdu(self):
        if False:
            print('Hello World!')
        '\n        A round trip test of creating a FitsHDU, adding a FITS file to it,\n        writing the FitsHDU out as part of a new FITS file, and then reading\n        it and recovering the original FITS file.\n        '
        self._test_create_fitshdu(compression=False)

    def test_create_fitshdu_with_compression(self):
        if False:
            print('Hello World!')
        'Same as test_create_fitshdu but with gzip compression enabled.'
        self._test_create_fitshdu(compression=True)

    def test_create_fitshdu_from_filename(self):
        if False:
            return 10
        'Regression test on `FitsHDU.fromfile`'
        a = np.arange(100)
        phdu = fits.PrimaryHDU(data=a)
        phdu.header['TEST1'] = 'A'
        phdu.header['TEST2'] = 'B'
        imghdu = fits.ImageHDU(data=a + 1)
        phdu.header['TEST3'] = 'C'
        phdu.header['TEST4'] = 'D'
        hdul = fits.HDUList([phdu, imghdu])
        hdul.writeto(self.temp('test.fits'))
        fitshdu = fits.FitsHDU.fromfile(self.temp('test.fits'))
        hdul2 = fitshdu.hdulist
        assert len(hdul2) == 2
        assert fits.FITSDiff(hdul, hdul2).identical

    def _test_create_fitshdu(self, compression=False):
        if False:
            i = 10
            return i + 15
        hdul_orig = fits.open(self.data('test0.fits'), do_not_scale_image_data=True)
        fitshdu = fits.FitsHDU.fromhdulist(hdul_orig, compress=compression)
        hdul_orig.append(fitshdu)
        hdul_orig.writeto(self.temp('tmp.fits'), overwrite=True)
        del hdul_orig[-1]
        hdul = fits.open(self.temp('tmp.fits'))
        assert isinstance(hdul[-1], fits.FitsHDU)
        wrapped = hdul[-1].hdulist
        assert isinstance(wrapped, fits.HDUList)
        assert hdul_orig.info(output=False) == wrapped.info(output=False)
        assert (hdul[1].data == wrapped[1].data).all()
        assert (hdul[2].data == wrapped[2].data).all()
        assert (hdul[3].data == wrapped[3].data).all()
        assert (hdul[4].data == wrapped[4].data).all()
        hdul_orig.close()
        hdul.close()