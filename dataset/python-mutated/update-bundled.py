"""
This test checks whether the libraries we're bundling are out of date and need to be synced with
a newer upstream release.
"""
from __future__ import annotations
import fnmatch
import json
import re
import sys
from ansible.module_utils.compat.version import LooseVersion
import packaging.specifiers
from ansible.module_utils.urls import open_url
BUNDLED_RE = re.compile(b'\\b_BUNDLED_METADATA\\b')

def get_bundled_libs(paths):
    if False:
        i = 10
        return i + 15
    '\n    Return the set of known bundled libraries\n\n    :arg paths: The paths which the test has been instructed to check\n    :returns: The list of all files which we know to contain bundled libraries.  If a bundled\n        library consists of multiple files, this should be the file which has metadata included.\n    '
    bundled_libs = set()
    for filename in fnmatch.filter(paths, 'lib/ansible/compat/*/__init__.py'):
        bundled_libs.add(filename)
    bundled_libs.add('lib/ansible/module_utils/distro/__init__.py')
    bundled_libs.add('lib/ansible/module_utils/six/__init__.py')
    bundled_libs.add('lib/ansible/module_utils/urls.py')
    return bundled_libs

def get_files_with_bundled_metadata(paths):
    if False:
        i = 10
        return i + 15
    '\n    Search for any files which have bundled metadata inside of them\n\n    :arg paths: Iterable of filenames to search for metadata inside of\n    :returns: A set of pathnames which contained metadata\n    '
    with_metadata = set()
    for path in paths:
        with open(path, 'rb') as f:
            body = f.read()
        if BUNDLED_RE.search(body):
            with_metadata.add(path)
    return with_metadata

def get_bundled_metadata(filename):
    if False:
        for i in range(10):
            print('nop')
    "\n    Retrieve the metadata about a bundled library from a python file\n\n    :arg filename: The filename to look inside for the metadata\n    :raises ValueError: If we're unable to extract metadata from the file\n    :returns: The metadata from the python file\n    "
    with open(filename, 'r') as module:
        for line in module:
            if line.strip().startswith('# NOT_BUNDLED'):
                return None
            if line.strip().startswith('# CANT_UPDATE'):
                print('{0} marked as CANT_UPDATE, so skipping. Manual check for CVEs required.'.format(filename))
                return None
            if line.strip().startswith('_BUNDLED_METADATA'):
                data = line[line.index('{'):].strip()
                break
        else:
            raise ValueError('Unable to check bundled library for update.  Please add _BUNDLED_METADATA dictionary to the library file with information on pypi name and bundled version.')
        metadata = json.loads(data)
    return metadata

def get_latest_applicable_version(pypi_data, constraints=None):
    if False:
        while True:
            i = 10
    "Get the latest pypi version of the package that we allow\n\n    :arg pypi_data: Pypi information about the data as returned by\n        ``https://pypi.org/pypi/{pkg_name}/json``\n    :kwarg constraints: version constraints on what we're allowed to use as specified by\n        the bundled metadata\n    :returns: The most recent version on pypi that are allowed by ``constraints``\n    "
    latest_version = '0'
    if constraints:
        version_specification = packaging.specifiers.SpecifierSet(constraints)
        for version in pypi_data['releases']:
            if version in version_specification:
                if LooseVersion(version) > LooseVersion(latest_version):
                    latest_version = version
    else:
        latest_version = pypi_data['info']['version']
    return latest_version

def main():
    if False:
        for i in range(10):
            print('nop')
    'Entrypoint to the script'
    paths = sys.argv[1:] or sys.stdin.read().splitlines()
    bundled_libs = get_bundled_libs(paths)
    files_with_bundled_metadata = get_files_with_bundled_metadata(paths)
    for filename in files_with_bundled_metadata.difference(bundled_libs):
        if filename.startswith('test/support/'):
            continue
        print('{0}: ERROR: File contains _BUNDLED_METADATA but needs to be added to test/sanity/code-smell/update-bundled.py'.format(filename))
    for filename in bundled_libs:
        try:
            metadata = get_bundled_metadata(filename)
        except ValueError as e:
            print('{0}: ERROR: {1}'.format(filename, e))
            continue
        except (IOError, OSError) as e:
            if e.errno == 2:
                print('{0}: ERROR: {1}.  Perhaps the bundled library has been removed or moved and the bundled library test needs to be modified as well?'.format(filename, e))
        if metadata is None:
            continue
        pypi_fh = open_url('https://pypi.org/pypi/{0}/json'.format(metadata['pypi_name']))
        pypi_data = json.loads(pypi_fh.read().decode('utf-8'))
        constraints = metadata.get('version_constraints', None)
        latest_version = get_latest_applicable_version(pypi_data, constraints)
        if LooseVersion(metadata['version']) < LooseVersion(latest_version):
            print('{0}: UPDATE {1} from {2} to {3} {4}'.format(filename, metadata['pypi_name'], metadata['version'], latest_version, 'https://pypi.org/pypi/{0}/json'.format(metadata['pypi_name'])))
if __name__ == '__main__':
    main()