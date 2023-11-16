"""
Script to download NumPy wheels from the Anaconda staging area.

Usage::

    $ ./tools/download-wheels.py <version> -w <optional-wheelhouse>

The default wheelhouse is ``release/installers``.

Dependencies
------------

- beautifulsoup4
- urllib3

Examples
--------

While in the repository root::

    $ python tools/download-wheels.py 1.19.0
    $ python tools/download-wheels.py 1.19.0 -w ~/wheelhouse

"""
import os
import re
import shutil
import argparse
import urllib3
from bs4 import BeautifulSoup
__version__ = '0.1'
STAGING_URL = 'https://anaconda.org/multibuild-wheels-staging/numpy'
PREFIX = 'numpy'
WHL = '-.*\\.whl$'
ZIP = '\\.zip$'
GZIP = '\\.tar\\.gz$'
SUFFIX = f'({WHL}|{GZIP}|{ZIP})'

def get_wheel_names(version):
    if False:
        print('Hello World!')
    ' Get wheel names from Anaconda HTML directory.\n\n    This looks in the Anaconda multibuild-wheels-staging page and\n    parses the HTML to get all the wheel names for a release version.\n\n    Parameters\n    ----------\n    version : str\n        The release version. For instance, "1.18.3".\n\n    '
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')
    tmpl = re.compile(f'^.*{PREFIX}-{version}{SUFFIX}')
    index_url = f'{STAGING_URL}/files'
    index_html = http.request('GET', index_url)
    soup = BeautifulSoup(index_html.data, 'html.parser')
    return soup.find_all(string=tmpl)

def download_wheels(version, wheelhouse):
    if False:
        for i in range(10):
            print('nop')
    'Download release wheels.\n\n    The release wheels for the given NumPy version are downloaded\n    into the given directory.\n\n    Parameters\n    ----------\n    version : str\n        The release version. For instance, "1.18.3".\n    wheelhouse : str\n        Directory in which to download the wheels.\n\n    '
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')
    wheel_names = get_wheel_names(version)
    for (i, wheel_name) in enumerate(wheel_names):
        wheel_url = f'{STAGING_URL}/{version}/download/{wheel_name}'
        wheel_path = os.path.join(wheelhouse, wheel_name)
        with open(wheel_path, 'wb') as f:
            with http.request('GET', wheel_url, preload_content=False) as r:
                print(f'{i + 1:<4}{wheel_name}')
                shutil.copyfileobj(r, f)
    print(f'\nTotal files downloaded: {len(wheel_names)}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='NumPy version to download.')
    parser.add_argument('-w', '--wheelhouse', default=os.path.join(os.getcwd(), 'release', 'installers'), help='Directory in which to store downloaded wheels\n[defaults to <cwd>/release/installers]')
    args = parser.parse_args()
    wheelhouse = os.path.expanduser(args.wheelhouse)
    if not os.path.isdir(wheelhouse):
        raise RuntimeError(f"{wheelhouse} wheelhouse directory is not present. Perhaps you need to use the '-w' flag to specify one.")
    download_wheels(args.version, wheelhouse)