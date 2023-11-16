import pytest
from astropy import __version__ as version
from astropy.io.fits.scripts import fitsinfo
from .conftest import FitsTestCase

class TestFitsinfo(FitsTestCase):

    def test_help(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(SystemExit) as e:
            fitsinfo.main(['-h'])
        assert e.value.code == 0

    def test_version(self, capsys):
        if False:
            while True:
                i = 10
        with pytest.raises(SystemExit) as e:
            fitsinfo.main(['--version'])
            out = capsys.readouterr()[0]
            assert out == f'fitsinfo {version}'
        assert e.value.code == 0

    def test_onefile(self, capsys):
        if False:
            for i in range(10):
                print('nop')
        fitsinfo.main([self.data('arange.fits')])
        (out, err) = capsys.readouterr()
        out = out.splitlines()
        assert len(out) == 3
        assert out[1].startswith('No.    Name      Ver    Type      Cards   Dimensions   Format')
        assert out[2].startswith('  0  PRIMARY       1 PrimaryHDU       7   (11, 10, 7)   int32')

    def test_multiplefiles(self, capsys):
        if False:
            print('Hello World!')
        fitsinfo.main([self.data('arange.fits'), self.data('ascii.fits')])
        (out, err) = capsys.readouterr()
        out = out.splitlines()
        assert len(out) == 8
        assert out[1].startswith('No.    Name      Ver    Type      Cards   Dimensions   Format')
        assert out[2].startswith('  0  PRIMARY       1 PrimaryHDU       7   (11, 10, 7)   int32')
        assert out[3] == ''
        assert out[7].startswith('  1                1 TableHDU        20   5R x 2C   [E10.4, I5]')