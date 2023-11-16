"""
Windows registry functions for Microsoft Visual C/C++.
"""
import os
from SCons.Util import HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER, RegGetValue
from ..common import debug
from . import Util
from . import Dispatcher
Dispatcher.register_modulename(__name__)
REG_EXPAND_SZ = 2

def read_value(hkey, subkey_valname, expand=True):
    if False:
        for i in range(10):
            print('nop')
    try:
        rval_t = RegGetValue(hkey, subkey_valname)
    except OSError:
        debug('OSError: hkey=%s, subkey=%s', repr(hkey), repr(subkey_valname))
        return None
    (rval, regtype) = rval_t
    if regtype == REG_EXPAND_SZ and expand:
        rval = os.path.expandvars(rval)
    debug('hkey=%s, subkey=%s, rval=%s', repr(hkey), repr(subkey_valname), repr(rval))
    return rval

def registry_query_path(key, val, suffix, expand=True):
    if False:
        while True:
            i = 10
    extval = val + '\\' + suffix if suffix else val
    qpath = read_value(key, extval, expand=expand)
    if qpath and os.path.exists(qpath):
        qpath = Util.process_path(qpath)
    else:
        qpath = None
    return (qpath, key, val, extval)
REG_SOFTWARE_MICROSOFT = [(HKEY_LOCAL_MACHINE, 'Software\\Wow6432Node\\Microsoft'), (HKEY_CURRENT_USER, 'Software\\Wow6432Node\\Microsoft'), (HKEY_LOCAL_MACHINE, 'Software\\Microsoft'), (HKEY_CURRENT_USER, 'Software\\Microsoft')]

def microsoft_query_paths(suffix, usrval=None, expand=True):
    if False:
        print('Hello World!')
    paths = []
    records = []
    for (key, val) in REG_SOFTWARE_MICROSOFT:
        extval = val + '\\' + suffix if suffix else val
        qpath = read_value(key, extval, expand=expand)
        if qpath and os.path.exists(qpath):
            qpath = Util.process_path(qpath)
            if qpath not in paths:
                paths.append(qpath)
                records.append((qpath, key, val, extval, usrval))
    return records

def microsoft_query_keys(suffix, usrval=None, expand=True):
    if False:
        return 10
    records = []
    for (key, val) in REG_SOFTWARE_MICROSOFT:
        extval = val + '\\' + suffix if suffix else val
        rval = read_value(key, extval, expand=expand)
        if rval:
            records.append((rval, key, val, extval, usrval))
    return records

def microsoft_sdks(version):
    if False:
        for i in range(10):
            print('nop')
    return '\\'.join(['Microsoft SDKs\\Windows', 'v' + version, 'InstallationFolder'])

def sdk_query_paths(version):
    if False:
        print('Hello World!')
    q = microsoft_sdks(version)
    return microsoft_query_paths(q)

def windows_kits(version):
    if False:
        for i in range(10):
            print('nop')
    return 'Windows Kits\\Installed Roots\\KitsRoot' + version

def windows_kit_query_paths(version):
    if False:
        i = 10
        return i + 15
    q = windows_kits(version)
    return microsoft_query_paths(q)

def vstudio_sxs_vc7(version):
    if False:
        for i in range(10):
            print('nop')
    return '\\'.join(['VisualStudio\\SxS\\VC7', version])

def devdiv_vs_servicing_component(version, component):
    if False:
        i = 10
        return i + 15
    return '\\'.join(['DevDiv\\VS\\Servicing', version, component, 'Install'])