from astropy.nddata.nddata_base import NDDataBase

class MinimalSubclass(NDDataBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

    @property
    def data(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def mask(self):
        if False:
            for i in range(10):
                print('nop')
        return super().mask

    @property
    def unit(self):
        if False:
            return 10
        return super().unit

    @property
    def wcs(self):
        if False:
            i = 10
            return i + 15
        return super().wcs

    @property
    def meta(self):
        if False:
            return 10
        return super().meta

    @property
    def uncertainty(self):
        if False:
            i = 10
            return i + 15
        return super().uncertainty

    @property
    def psf(self):
        if False:
            i = 10
            return i + 15
        return super().psf

class MinimalSubclassNoPSF(NDDataBase):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @property
    def data(self):
        if False:
            return 10
        return None

    @property
    def mask(self):
        if False:
            return 10
        return super().mask

    @property
    def unit(self):
        if False:
            print('Hello World!')
        return super().unit

    @property
    def wcs(self):
        if False:
            for i in range(10):
                print('nop')
        return super().wcs

    @property
    def meta(self):
        if False:
            while True:
                i = 10
        return super().meta

    @property
    def uncertainty(self):
        if False:
            return 10
        return super().uncertainty

def test_nddata_base_subclass():
    if False:
        print('Hello World!')
    a = MinimalSubclass()
    assert a.meta is None
    assert a.data is None
    assert a.mask is None
    assert a.unit is None
    assert a.wcs is None
    assert a.uncertainty is None
    assert a.psf is None

def test_omitting_psf_is_ok():
    if False:
        while True:
            i = 10
    b = MinimalSubclassNoPSF()
    assert b.psf is None