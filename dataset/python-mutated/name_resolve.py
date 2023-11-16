"""
This module contains convenience functions for getting a coordinate object
for a named object by querying SESAME and getting the first returned result.
Note that this is intended to be a convenience, and is very simple. If you
need precise coordinates for an object you should find the appropriate
reference for that measurement and input the coordinates manually.
"""
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from astropy import units as u
from astropy.utils import data
from astropy.utils.data import download_file, get_file_contents
from astropy.utils.state import ScienceState
from .sky_coordinate import SkyCoord
__all__ = ['get_icrs_coordinates']

class sesame_url(ScienceState):
    """
    The URL(s) to Sesame's web-queryable database.
    """
    _value = ['https://cds.unistra.fr/cgi-bin/nph-sesame/', 'http://vizier.cfa.harvard.edu/viz-bin/nph-sesame/']

    @classmethod
    def validate(cls, value):
        if False:
            return 10
        return value

class sesame_database(ScienceState):
    """
    This specifies the default database that SESAME will query when
    using the name resolve mechanism in the coordinates
    subpackage. Default is to search all databases, but this can be
    'all', 'simbad', 'ned', or 'vizier'.
    """
    _value = 'all'

    @classmethod
    def validate(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if value not in ['all', 'simbad', 'ned', 'vizier']:
            raise ValueError(f"Unknown database '{value}'")
        return value

class NameResolveError(Exception):
    pass

def _parse_response(resp_data):
    if False:
        i = 10
        return i + 15
    '\n    Given a string response from SESAME, parse out the coordinates by looking\n    for a line starting with a J, meaning ICRS J2000 coordinates.\n\n    Parameters\n    ----------\n    resp_data : str\n        The string HTTP response from SESAME.\n\n    Returns\n    -------\n    ra : str\n        The string Right Ascension parsed from the HTTP response.\n    dec : str\n        The string Declination parsed from the HTTP response.\n    '
    pattr = re.compile('%J\\s*([0-9\\.]+)\\s*([\\+\\-\\.0-9]+)')
    matched = pattr.search(resp_data)
    if matched is None:
        return (None, None)
    else:
        (ra, dec) = matched.groups()
        return (ra, dec)

def get_icrs_coordinates(name, parse=False, cache=False):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve an ICRS object by using an online name resolving service to\n    retrieve coordinates for the specified name. By default, this will\n    search all available databases until a match is found. If you would like\n    to specify the database, use the science state\n    ``astropy.coordinates.name_resolve.sesame_database``. You can also\n    specify a list of servers to use for querying Sesame using the science\n    state ``astropy.coordinates.name_resolve.sesame_url``. This will try\n    each one in order until a valid response is returned. By default, this\n    list includes the main Sesame host and a mirror at vizier.  The\n    configuration item `astropy.utils.data.Conf.remote_timeout` controls the\n    number of seconds to wait for a response from the server before giving\n    up.\n\n    Parameters\n    ----------\n    name : str\n        The name of the object to get coordinates for, e.g. ``\'M42\'``.\n    parse : bool\n        Whether to attempt extracting the coordinates from the name by\n        parsing with a regex. For objects catalog names that have\n        J-coordinates embedded in their names eg:\n        \'CRTS SSS100805 J194428-420209\', this may be much faster than a\n        sesame query for the same object name. The coordinates extracted\n        in this way may differ from the database coordinates by a few\n        deci-arcseconds, so only use this option if you do not need\n        sub-arcsecond accuracy for coordinates.\n    cache : bool, str, optional\n        Determines whether to cache the results or not. Passed through to\n        `~astropy.utils.data.download_file`, so pass "update" to update the\n        cached value.\n\n    Returns\n    -------\n    coord : `astropy.coordinates.ICRS` object\n        The object\'s coordinates in the ICRS frame.\n\n    '
    if parse:
        from . import jparser
        if jparser.search(name):
            return jparser.to_skycoord(name)
        else:
            pass
    database = sesame_database.get()
    db = database.upper()[0]
    urls = []
    domains = []
    for url in sesame_url.get():
        domain = urllib.parse.urlparse(url).netloc
        if domain not in domains:
            domains.append(domain)
            fmt_url = os.path.join(url, '{db}?{name}')
            fmt_url = fmt_url.format(name=urllib.parse.quote(name), db=db)
            urls.append(fmt_url)
    exceptions = []
    for url in urls:
        try:
            resp_data = get_file_contents(download_file(url, cache=cache, show_progress=False))
            break
        except urllib.error.URLError as e:
            exceptions.append(e)
            continue
        except socket.timeout as e:
            e.reason = f'Request took longer than the allowed {data.conf.remote_timeout:.1f} seconds'
            exceptions.append(e)
            continue
    else:
        messages = [f'{url}: {e.reason}' for (url, e) in zip(urls, exceptions)]
        raise NameResolveError(f'All Sesame queries failed. Unable to retrieve coordinates. See errors per URL below: \n {os.linesep.join(messages)}')
    (ra, dec) = _parse_response(resp_data)
    if ra is None or dec is None:
        if db == 'A':
            err = f"Unable to find coordinates for name '{name}' using {url}"
        else:
            err = f"Unable to find coordinates for name '{name}' in database {database} using {url}"
        raise NameResolveError(err)
    sc = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    return sc