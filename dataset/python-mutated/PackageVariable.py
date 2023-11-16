"""engine.SCons.Variables.PackageVariable

This file defines the option type for SCons implementing 'package
activation'.

To be used whenever a 'package' may be enabled/disabled and the
package path may be specified.

Usage example:

  Examples:
      x11=no   (disables X11 support)
      x11=yes  (will search for the package installation dir)
      x11=/usr/local/X11 (will check this path for existence)

  To replace autoconf's --with-xxx=yyy ::

      opts = Variables()
      opts.Add(PackageVariable('x11',
                             'use X11 installed here (yes = search some places',
                             'yes'))
      ...
      if env['x11'] == True:
          dir = ... search X11 in some standard places ...
          env['x11'] = dir
      if env['x11']:
          ... build with x11 ...
"""
__revision__ = 'src/engine/SCons/Variables/PackageVariable.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__all__ = ['PackageVariable']
import SCons.Errors
__enable_strings = ('1', 'yes', 'true', 'on', 'enable', 'search')
__disable_strings = ('0', 'no', 'false', 'off', 'disable')

def _converter(val):
    if False:
        return 10
    '\n    '
    lval = val.lower()
    if lval in __enable_strings:
        return True
    if lval in __disable_strings:
        return False
    return val

def _validator(key, val, env, searchfunc):
    if False:
        return 10
    '\n    '
    import os
    if env[key] is True:
        if searchfunc:
            env[key] = searchfunc(key, val)
    elif env[key] and (not os.path.exists(val)):
        raise SCons.Errors.UserError('Path does not exist for option %s: %s' % (key, val))

def PackageVariable(key, help, default, searchfunc=None):
    if False:
        return 10
    "\n    The input parameters describe a 'package list' option, thus they\n    are returned with the correct converter and validator appended. The\n    result is usable for input to opts.Add() .\n\n    A 'package list' option may either be 'all', 'none' or a list of\n    package names (separated by space).\n    "
    help = '\n    '.join((help, '( yes | no | /path/to/%s )' % key))
    return (key, help, default, lambda k, v, e: _validator(k, v, e, searchfunc), _converter)