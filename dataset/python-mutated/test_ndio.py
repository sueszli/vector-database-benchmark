from astropy.nddata import NDData, NDDataRef, NDIOMixin
NDDataIO = NDDataRef

def test_simple_write_read():
    if False:
        while True:
            i = 10
    ndd = NDDataIO([1, 2, 3])
    assert hasattr(ndd, 'read')
    assert hasattr(ndd, 'write')