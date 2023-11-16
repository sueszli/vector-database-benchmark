import platform
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from astropy.io import fits
from .conftest import FitsTestCase

class TestUintFunctions(FitsTestCase):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.utypes = ('u2', 'u4', 'u8')
        cls.utype_map = {'u2': np.uint16, 'u4': np.uint32, 'u8': np.uint64}
        cls.itype_map = {'u2': np.int16, 'u4': np.int32, 'u8': np.int64}
        cls.format_map = {'u2': 'I', 'u4': 'J', 'u8': 'K'}

    @pytest.mark.parametrize(('utype', 'compressed'), [('u2', False), ('u4', False), ('u8', False), ('u2', True), ('u4', True)])
    def test_uint(self, utype, compressed):
        if False:
            print('Hello World!')
        bits = 8 * int(utype[1])
        if platform.architecture()[0] == '64bit' or bits != 64:
            if compressed:
                hdu = fits.CompImageHDU(np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64))
                hdu_number = 1
            else:
                hdu = fits.PrimaryHDU(np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64))
                hdu_number = 0
            hdu.scale(f'int{bits:d}', '', bzero=2 ** (bits - 1))
            hdu.writeto(self.temp('tempfile.fits'), overwrite=True)
            with fits.open(self.temp('tempfile.fits'), uint=True) as hdul:
                assert hdul[hdu_number].data.dtype == self.utype_map[utype]
                assert (hdul[hdu_number].data == np.array([2 ** bits - 3, 2 ** bits - 2, 2 ** bits - 1, 0, 1, 2, 3], dtype=self.utype_map[utype])).all()
                hdul.writeto(self.temp('tempfile1.fits'))
                with fits.open(self.temp('tempfile1.fits'), uint16=True) as hdul1:
                    d1 = hdul[hdu_number].data
                    d2 = hdul1[hdu_number].data
                    assert (d1 == d2).all()
                    if not compressed:
                        sec = hdul[hdu_number].section[:1]
                        assert sec.dtype.name == f'uint{bits}'
                        assert (sec == d1[:1]).all()

    @pytest.mark.parametrize('utype', ('u2', 'u4', 'u8'))
    def test_uint_columns(self, utype):
        if False:
            for i in range(10):
                print('nop')
        'Test basic functionality of tables with columns containing\n        pseudo-unsigned integers.  See\n        https://github.com/astropy/astropy/pull/906\n        '
        bits = 8 * int(utype[1])
        if platform.architecture()[0] == '64bit' or bits != 64:
            bzero = self.utype_map[utype](2 ** (bits - 1))
            one = self.utype_map[utype](1)
            u0 = np.arange(bits + 1, dtype=self.utype_map[utype])
            u = 2 ** u0 - one
            if bits == 64:
                u[63] = bzero - one
                u[64] = u[63] + u[63] + one
            uu = (u - bzero).view(self.itype_map[utype])
            col = fits.Column(name=utype, array=u, format=self.format_map[utype], bzero=bzero)
            table = fits.BinTableHDU.from_columns([col])
            assert (table.data[utype] == u).all()
            assert (table.data.base.base[utype] == uu).all()
            hdu0 = fits.PrimaryHDU()
            hdulist = fits.HDUList([hdu0, table])
            hdulist.writeto(self.temp('tempfile.fits'), overwrite=True)
            del hdulist
            with fits.open(self.temp('tempfile.fits'), uint=True) as hdulist2:
                hdudata = hdulist2[1].data
                assert (hdudata[utype] == u).all()
                assert hdudata[utype].dtype == self.utype_map[utype]
                assert (hdudata.base[utype] == uu).all()
            v = u.view(dtype=[(utype, self.utype_map[utype])])
            fits.writeto(self.temp('tempfile2.fits'), v, overwrite=True)
            with fits.open(self.temp('tempfile2.fits'), uint=True) as hdulist3:
                hdudata3 = hdulist3[1].data
                assert (hdudata3.base[utype] == table.data.base.base[utype]).all()
                assert (hdudata3[utype] == table.data[utype]).all()
                assert (hdudata3[utype] == u).all()

    def test_uint_slice(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fix for https://github.com/astropy/astropy/issues/5490\n        if data is sliced first, make sure the data is still converted as uint\n        '
        dataref = np.arange(2 ** 16, dtype=np.uint16)
        tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='a', format='I', array=np.arange(2 ** 16, dtype=np.int16)), fits.Column(name='b', format='I', bscale=1, bzero=2 ** 15, array=dataref)])
        tbhdu.writeto(self.temp('test_scaled_slicing.fits'))
        with fits.open(self.temp('test_scaled_slicing.fits')) as hdulist:
            data = hdulist[1].data
        assert_array_equal(data['b'], dataref)
        sel = data['a'] >= 0
        assert_array_equal(data[sel]['b'], dataref[sel])
        assert data[sel]['b'].dtype == dataref[sel].dtype
        with fits.open(self.temp('test_scaled_slicing.fits')) as hdulist:
            data = hdulist[1].data
        assert_array_equal(data[sel]['b'], dataref[sel])
        assert data[sel]['b'].dtype == dataref[sel].dtype