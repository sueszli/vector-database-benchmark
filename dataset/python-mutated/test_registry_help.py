from io import StringIO
from astropy.nddata import CCDData
from astropy.table import Table

def test_table_read_help_fits():
    if False:
        i = 10
        return i + 15
    "\n    Test dynamically created documentation help via the I/O registry for 'fits'.\n    "
    out = StringIO()
    Table.read.help('fits', out)
    doc = out.getvalue()
    assert 'Table.read general documentation' not in doc
    assert 'The available built-in formats' not in doc
    assert "Table.read(format='fits') documentation" in doc
    assert 'hdu : int or str, optional' in doc

def test_table_read_help_ascii():
    if False:
        while True:
            i = 10
    "\n    Test dynamically created documentation help via the I/O registry for 'ascii'.\n    "
    out = StringIO()
    Table.read.help('ascii', out)
    doc = out.getvalue()
    assert 'Table.read general documentation' not in doc
    assert 'The available built-in formats' not in doc
    assert "Table.read(format='ascii') documentation" in doc
    assert 'delimiter : str' in doc
    assert "ASCII reader 'ascii' details" in doc
    assert 'Character-delimited table with a single header line' in doc

def test_table_write_help_hdf5():
    if False:
        print('Hello World!')
    "\n    Test dynamically created documentation help via the I/O registry for 'hdf5'.\n    "
    out = StringIO()
    Table.write.help('hdf5', out)
    doc = out.getvalue()
    assert 'Table.write general documentation' not in doc
    assert 'The available built-in formats' not in doc
    assert "Table.write(format='hdf5') documentation" in doc
    assert 'Write a Table object to an HDF5 file' in doc
    assert 'compression : bool or str or int' in doc

def test_list_formats():
    if False:
        return 10
    '\n    Test getting list of available formats\n    '
    out = StringIO()
    CCDData.write.list_formats(out)
    output = out.getvalue()
    assert output == 'Format Read Write Auto-identify\n------ ---- ----- -------------\n  fits  Yes   Yes           Yes'

def test_table_write_help_fits():
    if False:
        return 10
    "\n    Test dynamically created documentation help via the I/O registry for 'fits'.\n    "
    out = StringIO()
    Table.write.help('fits', out)
    doc = out.getvalue()
    assert 'Table.write general documentation' not in doc
    assert 'The available built-in formats' not in doc
    assert "Table.write(format='fits') documentation" in doc
    assert 'Write a Table object to a FITS file' in doc

def test_table_write_help_no_format():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test dynamically created documentation help via the I/O registry for no\n    format provided.\n    '
    out = StringIO()
    Table.write.help(out=out)
    doc = out.getvalue()
    assert 'Table.write general documentation' in doc
    assert 'The available built-in formats' in doc

def test_table_read_help_no_format():
    if False:
        print('Hello World!')
    '\n    Test dynamically created documentation help via the I/O registry for not\n    format provided.\n    '
    out = StringIO()
    Table.read.help(out=out)
    doc = out.getvalue()
    assert 'Table.read general documentation' in doc
    assert 'The available built-in formats' in doc

def test_ccddata_write_help_fits():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test dynamically created documentation help via the I/O registry for 'fits'.\n    "
    out = StringIO()
    CCDData.write.help('fits', out)
    doc = out.getvalue()
    assert "CCDData.write(format='fits') documentation" in doc
    assert 'Write CCDData object to FITS file' in doc
    assert 'key_uncertainty_type : str, optional' in doc

def test_ccddata_read_help_fits():
    if False:
        i = 10
        return i + 15
    "Test dynamically created documentation help via the I/O registry for\n    CCDData 'fits'.\n\n    "
    out = StringIO()
    CCDData.read.help('fits', out)
    doc = out.getvalue()
    assert "CCDData.read(format='fits') documentation" in doc
    assert 'Generate a CCDData object from a FITS file' in doc
    assert 'hdu_uncertainty : str or None, optional' in doc

def test_table_write_help_jsviewer():
    if False:
        return 10
    "\n    Test dynamically created documentation help via the I/O registry for\n    'jsviewer'.\n    "
    out = StringIO()
    Table.write.help('jsviewer', out)
    doc = out.getvalue()
    assert 'Table.write general documentation' not in doc
    assert 'The available built-in formats' not in doc
    assert "Table.write(format='jsviewer') documentation" in doc