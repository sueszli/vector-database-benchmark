import pytest
from astropy.wcs.wcsapi.low_level_api import validate_physical_types

def test_validate_physical_types():
    if False:
        while True:
            i = 10
    validate_physical_types(['pos.eq.ra', 'pos.eq.ra'])
    validate_physical_types(['spect.dopplerVeloc.radio', 'custom:spam'])
    validate_physical_types(['time', None])
    with pytest.raises(ValueError, match="'Pos\\.eq\\.dec' is not a valid IOVA UCD1\\+ physical type"):
        validate_physical_types(['pos.eq.ra', 'Pos.eq.dec'])
    with pytest.raises(ValueError, match="'spam' is not a valid IOVA UCD1\\+ physical type"):
        validate_physical_types(['spam'])