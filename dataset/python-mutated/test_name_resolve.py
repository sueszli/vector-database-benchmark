"""
This module contains tests for the name resolve convenience module.
"""
import time
import urllib.request
import numpy as np
import pytest
from pytest_remotedata.disable_internet import no_internet
from astropy import units as u
from astropy.config import paths
from astropy.coordinates.name_resolve import NameResolveError, _parse_response, get_icrs_coordinates, sesame_database, sesame_url
from astropy.coordinates.sky_coordinate import SkyCoord
_cached_ngc3642 = dict()
_cached_ngc3642['simbad'] = '# NGC 3642    #Q22523669\n#=S=Simbad (via url):    1\n%@ 503952\n%I.0 NGC 3642\n%C.0 LIN\n%C.N0 15.15.01.00\n%J 170.5750583 +59.0742417 = 11:22:18.01 +59:04:27.2\n%V z 1593 0.005327 [0.000060] D 2002LEDA.........0P\n%D 1.673 1.657 75 (32767) (I) C 2006AJ....131.1163S\n%T 5 =32800000 D 2011A&A...532A..74B\n%#B 140\n\n\n#====Done (2013-Feb-12,16:37:11z)===='
_cached_ngc3642['vizier'] = '# NGC 3642    #Q22523677\n#=V=VizieR (local):    1\n%J 170.56 +59.08 = 11:22.2     +59:05\n%I.0 {NGC} 3642\n\n\n\n#====Done (2013-Feb-12,16:37:42z)===='
_cached_ngc3642['all'] = '# ngc3642    #Q22523722\n#=S=Simbad (via url):    1\n%@ 503952\n%I.0 NGC 3642\n%C.0 LIN\n%C.N0 15.15.01.00\n%J 170.5750583 +59.0742417 = 11:22:18.01 +59:04:27.2\n%V z 1593 0.005327 [0.000060] D 2002LEDA.........0P\n%D 1.673 1.657 75 (32767) (I) C 2006AJ....131.1163S\n%T 5 =32800000 D 2011A&A...532A..74B\n%#B 140\n\n\n#=V=VizieR (local):    1\n%J 170.56 +59.08 = 11:22.2     +59:05\n%I.0 {NGC} 3642\n\n\n#!N=NED : *** Could not access the server ***\n\n#====Done (2013-Feb-12,16:39:48z)===='
_cached_castor = dict()
_cached_castor['all'] = '# castor    #Q22524249\n#=S=Simbad (via url):    1\n%@ 983633\n%I.0 NAME CASTOR\n%C.0 **\n%C.N0 12.13.00.00\n%J 113.649471640 +31.888282216 = 07:34:35.87 +31:53:17.8\n%J.E [34.72 25.95 0] A 2007A&A...474..653V\n%P -191.45 -145.19 [3.95 2.95 0] A 2007A&A...474..653V\n%X 64.12 [3.75] A 2007A&A...474..653V\n%S A1V+A2Vm =0.0000D200.0030.0110000000100000 C 2001AJ....122.3466M\n%#B 179\n\n#!V=VizieR (local): No table found for: castor\n\n#!N=NED: ****object name not recognized by NED name interpreter\n#!N=NED: ***Not recognized by NED: castor\n\n\n\n#====Done (2013-Feb-12,16:52:02z)===='
_cached_castor['simbad'] = '# castor    #Q22524495\n#=S=Simbad (via url):    1\n%@ 983633\n%I.0 NAME CASTOR\n%C.0 **\n%C.N0 12.13.00.00\n%J 113.649471640 +31.888282216 = 07:34:35.87 +31:53:17.8\n%J.E [34.72 25.95 0] A 2007A&A...474..653V\n%P -191.45 -145.19 [3.95 2.95 0] A 2007A&A...474..653V\n%X 64.12 [3.75] A 2007A&A...474..653V\n%S A1V+A2Vm =0.0000D200.0030.0110000000100000 C 2001AJ....122.3466M\n%#B 179\n\n\n#====Done (2013-Feb-12,17:00:39z)===='

@pytest.mark.remote_data
def test_names():
    if False:
        for i in range(10):
            print('nop')
    if urllib.request.urlopen('https://cdsweb.unistra.fr/cgi-bin/nph-sesame').getcode() != 200:
        pytest.skip('SESAME appears to be down, skipping test_name_resolve.py:test_names()...')
    with pytest.raises(NameResolveError):
        get_icrs_coordinates('m87h34hhh')
    try:
        icrs = get_icrs_coordinates('NGC 3642')
    except NameResolveError:
        (ra, dec) = _parse_response(_cached_ngc3642['all'])
        icrs = SkyCoord(ra=float(ra) * u.degree, dec=float(dec) * u.degree)
    icrs_true = SkyCoord(ra='11h 22m 18.014s', dec='59d 04m 27.27s')
    np.testing.assert_almost_equal(icrs.ra.degree, icrs_true.ra.degree, 1)
    np.testing.assert_almost_equal(icrs.dec.degree, icrs_true.dec.degree, 1)
    try:
        icrs = get_icrs_coordinates('castor')
    except NameResolveError:
        (ra, dec) = _parse_response(_cached_castor['all'])
        icrs = SkyCoord(ra=float(ra) * u.degree, dec=float(dec) * u.degree)
    icrs_true = SkyCoord(ra='07h 34m 35.87s', dec='+31d 53m 17.8s')
    np.testing.assert_almost_equal(icrs.ra.degree, icrs_true.ra.degree, 1)
    np.testing.assert_almost_equal(icrs.dec.degree, icrs_true.dec.degree, 1)

@pytest.mark.remote_data
def test_name_resolve_cache(tmp_path):
    if False:
        print('Hello World!')
    from astropy.utils.data import get_cached_urls
    target_name = 'castor'
    (temp_cache_dir := (tmp_path / 'cache')).mkdir()
    with paths.set_temp_cache(temp_cache_dir, delete=True):
        assert len(get_cached_urls()) == 0
        icrs1 = get_icrs_coordinates(target_name, cache=True)
        urls = get_cached_urls()
        assert len(urls) == 1
        expected_urls = sesame_url.get()
        assert any((urls[0].startswith(x) for x in expected_urls)), f'{urls[0]} not in {expected_urls}'
        with no_internet():
            icrs2 = get_icrs_coordinates(target_name, cache=True)
        assert len(get_cached_urls()) == 1
        assert u.allclose(icrs1.ra, icrs2.ra)
        assert u.allclose(icrs1.dec, icrs2.dec)

def test_names_parse():
    if False:
        while True:
            i = 10
    test_names = ['CRTS SSS100805 J194428-420209', 'MASTER OT J061451.7-272535.5', '2MASS J06495091-0737408', '1RXS J042555.8-194534', 'SDSS J132411.57+032050.5', 'DENIS-P J203137.5-000511', '2QZ J142438.9-022739', 'CXOU J141312.3-652013']
    for name in test_names:
        sc = get_icrs_coordinates(name, parse=True)

@pytest.mark.remote_data
@pytest.mark.parametrize(('name', 'db_dict'), [('NGC 3642', _cached_ngc3642), ('castor', _cached_castor)])
def test_database_specify(name, db_dict):
    if False:
        print('Hello World!')
    for url in sesame_url.get():
        if urllib.request.urlopen(url).getcode() == 200:
            break
    else:
        pytest.skip('All SESAME mirrors appear to be down, skipping test_name_resolve.py:test_database_specify()...')
    for db in db_dict.keys():
        with sesame_database.set(db):
            icrs = SkyCoord.from_name(name)
        time.sleep(1)