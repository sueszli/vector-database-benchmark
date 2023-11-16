"""SCons.Tool.PharLapCommon

This module contains common code used by all Tools for the
Phar Lap ETS tool chain.  Right now, this is linkloc and
386asm.

"""
__revision__ = 'src/engine/SCons/Tool/PharLapCommon.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import os
import os.path
import SCons.Errors
import SCons.Util
import re

def getPharLapPath():
    if False:
        while True:
            i = 10
    'Reads the registry to find the installed path of the Phar Lap ETS\n    development kit.\n\n    Raises UserError if no installed version of Phar Lap can\n    be found.'
    if not SCons.Util.can_read_reg:
        raise SCons.Errors.InternalError('No Windows registry module was found')
    try:
        k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Pharlap\\ETS')
        (val, type) = SCons.Util.RegQueryValueEx(k, 'BaseDir')
        idx = val.find('\x00')
        if idx >= 0:
            val = val[:idx]
        return os.path.normpath(val)
    except SCons.Util.RegError:
        raise SCons.Errors.UserError('Cannot find Phar Lap ETS path in the registry.  Is it installed properly?')
REGEX_ETS_VER = re.compile('#define\\s+ETS_VER\\s+([0-9]+)')

def getPharLapVersion():
    if False:
        print('Hello World!')
    "Returns the version of the installed ETS Tool Suite as a\n    decimal number.  This version comes from the ETS_VER #define in\n    the embkern.h header.  For example, '#define ETS_VER 1010' (which\n    is what Phar Lap 10.1 defines) would cause this method to return\n    1010. Phar Lap 9.1 does not have such a #define, but this method\n    will return 910 as a default.\n\n    Raises UserError if no installed version of Phar Lap can\n    be found."
    include_path = os.path.join(getPharLapPath(), os.path.normpath('include/embkern.h'))
    if not os.path.exists(include_path):
        raise SCons.Errors.UserError('Cannot find embkern.h in ETS include directory.\nIs Phar Lap ETS installed properly?')
    with open(include_path, 'r') as f:
        mo = REGEX_ETS_VER.search(f.read())
    if mo:
        return int(mo.group(1))
    return 910

def addPharLapPaths(env):
    if False:
        for i in range(10):
            print('nop')
    'This function adds the path to the Phar Lap binaries, includes,\n    and libraries, if they are not already there.'
    ph_path = getPharLapPath()
    try:
        env_dict = env['ENV']
    except KeyError:
        env_dict = {}
        env['ENV'] = env_dict
    SCons.Util.AddPathIfNotExists(env_dict, 'PATH', os.path.join(ph_path, 'bin'))
    SCons.Util.AddPathIfNotExists(env_dict, 'INCLUDE', os.path.join(ph_path, 'include'))
    SCons.Util.AddPathIfNotExists(env_dict, 'LIB', os.path.join(ph_path, 'lib'))
    SCons.Util.AddPathIfNotExists(env_dict, 'LIB', os.path.join(ph_path, os.path.normpath('lib/vclib')))
    env['PHARLAP_PATH'] = getPharLapPath()
    env['PHARLAP_VERSION'] = str(getPharLapVersion())