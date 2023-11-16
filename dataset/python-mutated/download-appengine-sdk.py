"""Fetches the most recent GAE SDK and extracts it to the given directory."""
from __future__ import print_function
import os
import re
import sys
import zipfile
import requests
if sys.version_info[0] == 2:
    from StringIO import StringIO
elif sys.version_info[0] == 3:
    from io import StringIO
SDK_RELEASES_URL = 'https://www.googleapis.com/storage/v1/b/appengine-sdks/o?prefix=featured'
PYTHON_RELEASE_RE = re.compile('featured/google_appengine_(\\d+?)\\.(\\d+?)\\.(\\d+?)\\.zip')
SDK_RELEASE_RE = re.compile('release: \\"(\\d+?)\\.(\\d+?)\\.(\\d+?)\\"')

def get_gae_versions():
    if False:
        while True:
            i = 10
    'Gets a list of all of the available Python SDK versions, sorted with\n    the newest last.'
    r = requests.get(SDK_RELEASES_URL)
    r.raise_for_status()
    releases = r.json().get('items', {})
    versions_and_urls = []
    for release in releases:
        match = PYTHON_RELEASE_RE.match(release['name'])
        if not match:
            continue
        versions_and_urls.append(([int(x) for x in match.groups()], release['mediaLink']))
    return sorted(versions_and_urls, key=lambda x: x[0])

def is_existing_up_to_date(destination, latest_version):
    if False:
        while True:
            i = 10
    'Returns False if there is no existing install or if the existing install\n    is out of date. Otherwise, returns True.'
    version_path = os.path.join(destination, 'google_appengine', 'VERSION')
    if not os.path.exists(version_path):
        return False
    with open(version_path, 'r') as f:
        version_line = f.readline()
        match = SDK_RELEASE_RE.match(version_line)
        if not match:
            print('Unable to parse version from:', version_line)
            return False
        version = [int(x) for x in match.groups()]
    return version >= latest_version

def download_sdk(url):
    if False:
        print('Hello World!')
    'Downloads the SDK and returns a file-like object for the zip content.'
    r = requests.get(url)
    r.raise_for_status()
    return StringIO(r.content)

def extract_zip(zip, destination):
    if False:
        print('Hello World!')
    zip_contents = zipfile.ZipFile(zip)
    if not os.path.exists(destination):
        os.makedirs(destination)
    zip_contents.extractall(destination)

def fixup_version(destination, version):
    if False:
        i = 10
        return i + 15
    'Newer releases of the SDK do not have the version number set correctly\n    in the VERSION file. Fix it up.'
    version_path = os.path.join(destination, 'google_appengine', 'VERSION')
    with open(version_path, 'r') as f:
        version_data = f.read()
    version_data = version_data.replace('release: "0.0.0"', 'release: "{}"'.format('.'.join((str(x) for x in version))))
    with open(version_path, 'w') as f:
        f.write(version_data)

def download_command(destination):
    if False:
        for i in range(10):
            print('nop')
    'Downloads and extracts the latest App Engine SDK to the given\n    destination.'
    latest_two_versions = list(reversed(get_gae_versions()))[:2]
    zip = None
    version_number = None
    for version in latest_two_versions:
        if is_existing_up_to_date(destination, version[0]):
            print('App Engine SDK already exists and is up to date at {}.'.format(destination))
            return
        try:
            print('Downloading App Engine SDK {}'.format('.'.join([str(x) for x in version[0]])))
            zip = download_sdk(version[1])
            version_number = version[0]
            break
        except Exception as e:
            print('Failed to download: {}'.format(e))
            continue
    if not zip:
        return
    print('Extracting SDK to {}'.format(destination))
    extract_zip(zip, destination)
    fixup_version(destination, version_number)
    print('App Engine SDK installed.')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('download-appengine-sdk.py dest')
        sys.exit(1)
    download_command(sys.argv[1])