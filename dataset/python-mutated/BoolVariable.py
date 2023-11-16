"""engine.SCons.Variables.BoolVariable

This file defines the option type for SCons implementing true/false values.

Usage example::

    opts = Variables()
    opts.Add(BoolVariable('embedded', 'build for an embedded system', 0))
    ...
    if env['embedded'] == 1:
    ...
"""
__revision__ = 'src/engine/SCons/Variables/BoolVariable.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__all__ = ['BoolVariable']
import SCons.Errors
__true_strings = ('y', 'yes', 'true', 't', '1', 'on', 'all')
__false_strings = ('n', 'no', 'false', 'f', '0', 'off', 'none')

def _text2bool(val):
    if False:
        print('Hello World!')
    "\n    Converts strings to True/False depending on the 'truth' expressed by\n    the string. If the string can't be converted, the original value\n    will be returned.\n\n    See '__true_strings' and '__false_strings' for values considered\n    'true' or 'false respectively.\n\n    This is usable as 'converter' for SCons' Variables.\n    "
    lval = val.lower()
    if lval in __true_strings:
        return True
    if lval in __false_strings:
        return False
    raise ValueError('Invalid value for boolean option: %s' % val)

def _validator(key, val, env):
    if False:
        i = 10
        return i + 15
    "\n    Validates the given value to be either '0' or '1'.\n    \n    This is usable as 'validator' for SCons' Variables.\n    "
    if not env[key] in (True, False):
        raise SCons.Errors.UserError('Invalid value for boolean option %s: %s' % (key, env[key]))

def BoolVariable(key, help, default):
    if False:
        print('Hello World!')
    "\n    The input parameters describe a boolean option, thus they are\n    returned with the correct converter and validator appended. The\n    'help' text will by appended by '(yes|no) to show the valid\n    valued. The result is usable for input to opts.Add().\n    "
    return (key, '%s (yes|no)' % help, default, _validator, _text2bool)