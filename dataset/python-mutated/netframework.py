__revision__ = 'src/engine/SCons/Tool/MSCommon/netframework.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__doc__ = '\n'
import os
import re
import SCons.Util
from .common import read_reg, debug
_FRAMEWORKDIR_HKEY_ROOT = 'Software\\Microsoft\\.NETFramework\\InstallRoot'
_FRAMEWORKDIR_HKEY_ROOT = 'Software\\Microsoft\\Microsoft SDKs\\.NETFramework\\v2.0\\InstallationFolder'

def find_framework_root():
    if False:
        return 10
    try:
        froot = read_reg(_FRAMEWORKDIR_HKEY_ROOT)
        debug('Found framework install root in registry: {}'.format(froot))
    except SCons.Util.WinError as e:
        debug('Could not read reg key {}'.format(_FRAMEWORKDIR_HKEY_ROOT))
        return None
    if not os.path.exists(froot):
        debug('{} not found on fs'.format(froot))
        return None
    return froot

def query_versions():
    if False:
        while True:
            i = 10
    froot = find_framework_root()
    if froot:
        contents = os.listdir(froot)
        l = re.compile('v[0-9]+.*')
        versions = [e for e in contents if l.match(e)]

        def versrt(a, b):
            if False:
                i = 10
                return i + 15
            aa = a[1:]
            bb = b[1:]
            aal = aa.split('.')
            bbl = bb.split('.')
            return (aal > bbl) - (aal < bbl)
        versions.sort(versrt)
    else:
        versions = []
    return versions