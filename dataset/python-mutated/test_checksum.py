import numpy as np
from astropy.io import fits
from astropy.io.fits.tests.test_checksum import BaseChecksumTests

class TestChecksumFunctions(BaseChecksumTests):

    def test_compressed_image_data(self):
        if False:
            i = 10
            return i + 15
        with fits.open(self.data('comp.fits')) as h1:
            h1.writeto(self.temp('tmp.fits'), overwrite=True, checksum=True)
            with fits.open(self.temp('tmp.fits'), checksum=True) as h2:
                assert np.all(h1[1].data == h2[1].data)
                assert 'CHECKSUM' in h2[0].header
                assert h2[0].header['CHECKSUM'] == 'D8iBD6ZAD6fAD6ZA'
                assert 'DATASUM' in h2[0].header
                assert h2[0].header['DATASUM'] == '0'
                assert 'CHECKSUM' in h2[1].header
                assert h2[1].header['CHECKSUM'] == 'ZeAbdb8aZbAabb7a'
                assert 'DATASUM' in h2[1].header
                assert h2[1].header['DATASUM'] == '113055149'

    def test_failing_compressed_datasum(self):
        if False:
            i = 10
            return i + 15
        '\n        Regression test for https://github.com/astropy/astropy/issues/4587\n        '
        n = np.ones((10, 10), dtype='float32')
        comp_hdu = fits.CompImageHDU(n)
        comp_hdu.writeto(self.temp('tmp.fits'), checksum=True)
        with fits.open(self.temp('tmp.fits'), checksum=True) as hdul:
            assert np.all(hdul[1].data == comp_hdu.data)

    def test_compressed_image_data_int16(self):
        if False:
            return 10
        n = np.arange(100, dtype='int16')
        hdu = fits.ImageHDU(n)
        comp_hdu = fits.CompImageHDU(hdu.data, hdu.header)
        comp_hdu.writeto(self.temp('tmp.fits'), checksum=True)
        hdu.writeto(self.temp('uncomp.fits'), checksum=True)
        with fits.open(self.temp('tmp.fits'), checksum=True) as hdul:
            assert np.all(hdul[1].data == comp_hdu.data)
            assert np.all(hdul[1].data == hdu.data)
            assert 'CHECKSUM' in hdul[0].header
            assert hdul[0].header['CHECKSUM'] == 'D8iBD6ZAD6fAD6ZA'
            assert 'DATASUM' in hdul[0].header
            assert hdul[0].header['DATASUM'] == '0'
            assert 'CHECKSUM' in hdul[1].header
            assert hdul[1]._header['CHECKSUM'] == 'J5cCJ5c9J5cAJ5c9'
            assert 'DATASUM' in hdul[1].header
            assert hdul[1]._header['DATASUM'] == '2453673070'
            assert 'CHECKSUM' in hdul[1].header
            with fits.open(self.temp('uncomp.fits'), checksum=True) as hdul2:
                header_comp = hdul[1]._header
                header_uncomp = hdul2[1].header
                assert 'ZHECKSUM' in header_comp
                assert 'CHECKSUM' in header_uncomp
                assert header_uncomp['CHECKSUM'] == 'ZE94eE91ZE91bE91'
                assert header_comp['ZHECKSUM'] == header_uncomp['CHECKSUM']
                assert 'ZDATASUM' in header_comp
                assert 'DATASUM' in header_uncomp
                assert header_uncomp['DATASUM'] == '160565700'
                assert header_comp['ZDATASUM'] == header_uncomp['DATASUM']

    def test_compressed_image_data_float32(self):
        if False:
            print('Hello World!')
        n = np.arange(100, dtype='float32')
        hdu = fits.ImageHDU(n)
        comp_hdu = fits.CompImageHDU(hdu.data, hdu.header)
        comp_hdu.writeto(self.temp('tmp.fits'), checksum=True)
        hdu.writeto(self.temp('uncomp.fits'), checksum=True)
        with fits.open(self.temp('tmp.fits'), checksum=True) as hdul:
            assert np.all(hdul[1].data == comp_hdu.data)
            assert np.all(hdul[1].data == hdu.data)
            assert 'CHECKSUM' in hdul[0].header
            assert hdul[0].header['CHECKSUM'] == 'D8iBD6ZAD6fAD6ZA'
            assert 'DATASUM' in hdul[0].header
            assert hdul[0].header['DATASUM'] == '0'
            assert 'CHECKSUM' in hdul[1].header
            assert 'DATASUM' in hdul[1].header
            with fits.open(self.temp('uncomp.fits'), checksum=True) as hdul2:
                header_comp = hdul[1]._header
                header_uncomp = hdul2[1].header
                assert 'ZHECKSUM' in header_comp
                assert 'CHECKSUM' in header_uncomp
                assert header_uncomp['CHECKSUM'] == 'Cgr5FZo2Cdo2CZo2'
                assert header_comp['ZHECKSUM'] == header_uncomp['CHECKSUM']
                assert 'ZDATASUM' in header_comp
                assert 'DATASUM' in header_uncomp
                assert header_uncomp['DATASUM'] == '2393636889'
                assert header_comp['ZDATASUM'] == header_uncomp['DATASUM']