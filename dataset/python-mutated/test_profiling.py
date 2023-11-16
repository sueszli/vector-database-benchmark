import os
import numpy as np
import pytest
from astropy import wcs
from astropy.utils.data import get_pkg_data_contents, get_pkg_data_filenames
from astropy.utils.misc import NumpyRNGContext
from astropy.wcs.wcs import FITSFixedWarning
hdr_map_file_list = [os.path.basename(fname) for fname in get_pkg_data_filenames('data/maps', pattern='*.hdr')]

def test_read_map_files():
    if False:
        return 10
    n_map_files = 28
    assert len(hdr_map_file_list) == n_map_files, f'test_read_map_files has wrong number data files: found {len(hdr_map_file_list)}, expected  {n_map_files}'

@pytest.mark.parametrize('filename', hdr_map_file_list)
def test_map(filename):
    if False:
        while True:
            i = 10
    header = get_pkg_data_contents(os.path.join('data/maps', filename))
    wcsobj = wcs.WCS(header)
    with NumpyRNGContext(123456789):
        x = np.random.rand(2 ** 12, wcsobj.wcs.naxis)
        wcsobj.wcs_pix2world(x, 1)
        wcsobj.wcs_world2pix(x, 1)
hdr_spec_file_list = [os.path.basename(fname) for fname in get_pkg_data_filenames('data/spectra', pattern='*.hdr')]

def test_read_spec_files():
    if False:
        return 10
    n_spec_files = 6
    assert len(hdr_spec_file_list) == n_spec_files, f'test_spectra has wrong number data files: found {len(hdr_spec_file_list)}, expected  {n_spec_files}'

@pytest.mark.parametrize('filename', hdr_spec_file_list)
def test_spectrum(filename):
    if False:
        return 10
    header = get_pkg_data_contents(os.path.join('data', 'spectra', filename))
    with pytest.warns() as warning_lines:
        wcsobj = wcs.WCS(header)
    for w in warning_lines:
        assert issubclass(w.category, FITSFixedWarning)
    with NumpyRNGContext(123456789):
        x = np.random.rand(2 ** 16, wcsobj.wcs.naxis)
        wcsobj.wcs_pix2world(x, 1)
        wcsobj.wcs_world2pix(x, 1)