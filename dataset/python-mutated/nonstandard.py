import gzip
import io
from astropy.io.fits.file import _File
from astropy.io.fits.header import Header, _pad_length
from astropy.io.fits.util import fileobj_name
from astropy.utils import lazyproperty
from .base import NonstandardExtHDU
from .hdulist import HDUList

class FitsHDU(NonstandardExtHDU):
    """
    A non-standard extension HDU for encapsulating entire FITS files within a
    single HDU of a container FITS file.  These HDUs have an extension (that is
    an XTENSION keyword) of FITS.

    The FITS file contained in the HDU's data can be accessed by the `hdulist`
    attribute which returns the contained FITS file as an `HDUList` object.
    """
    _extension = 'FITS'

    @lazyproperty
    def hdulist(self):
        if False:
            i = 10
            return i + 15
        self._file.seek(self._data_offset)
        fileobj = io.BytesIO()
        fileobj.write(self._file.read(self.size))
        fileobj.seek(0)
        if self._header['COMPRESS']:
            fileobj = gzip.GzipFile(fileobj=fileobj)
        return HDUList.fromfile(fileobj, mode='readonly')

    @classmethod
    def fromfile(cls, filename, compress=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Like `FitsHDU.fromhdulist()`, but creates a FitsHDU from a file on\n        disk.\n\n        Parameters\n        ----------\n        filename : str\n            The path to the file to read into a FitsHDU\n        compress : bool, optional\n            Gzip compress the FITS file\n        '
        with HDUList.fromfile(filename) as hdulist:
            return cls.fromhdulist(hdulist, compress=compress)

    @classmethod
    def fromhdulist(cls, hdulist, compress=False):
        if False:
            print('Hello World!')
        '\n        Creates a new FitsHDU from a given HDUList object.\n\n        Parameters\n        ----------\n        hdulist : HDUList\n            A valid Headerlet object.\n        compress : bool, optional\n            Gzip compress the FITS file\n        '
        fileobj = bs = io.BytesIO()
        if compress:
            if hasattr(hdulist, '_file'):
                name = fileobj_name(hdulist._file)
            else:
                name = None
            fileobj = gzip.GzipFile(name, mode='wb', fileobj=bs)
        hdulist.writeto(fileobj)
        if compress:
            fileobj.close()
        padding = (_pad_length(bs.tell()) * cls._padding_byte).encode('ascii')
        bs.write(padding)
        bs.seek(0)
        cards = [('XTENSION', cls._extension, 'FITS extension'), ('BITPIX', 8, 'array data type'), ('NAXIS', 1, 'number of array dimensions'), ('NAXIS1', len(bs.getvalue()), 'Axis length'), ('PCOUNT', 0, 'number of parameters'), ('GCOUNT', 1, 'number of groups')]
        if len(hdulist) > 1:
            for (idx, hdu) in enumerate(hdulist[1:]):
                cards.append(('XIND' + str(idx + 1), hdu._header_offset, f'byte offset of extension {idx + 1}'))
        cards.append(('COMPRESS', compress, 'Uses gzip compression'))
        header = Header(cards)
        return cls._readfrom_internal(_File(bs), header=header)

    @classmethod
    def match_header(cls, header):
        if False:
            for i in range(10):
                print('nop')
        card = header.cards[0]
        if card.keyword != 'XTENSION':
            return False
        xtension = card.value
        if isinstance(xtension, str):
            xtension = xtension.rstrip()
        return xtension == cls._extension

    def _summary(self):
        if False:
            i = 10
            return i + 15
        return (self.name, self.ver, self.__class__.__name__, len(self._header))