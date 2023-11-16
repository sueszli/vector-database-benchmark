"""engine.SCons.Variables.EnumVariable

This file defines the option type for SCons allowing only specified
input-values.

Usage example::

    opts = Variables()
    opts.Add(EnumVariable('debug', 'debug output and symbols', 'no',
                      allowed_values=('yes', 'no', 'full'),
                      map={}, ignorecase=2))
    ...
    if env['debug'] == 'full':
    ...
"""
__revision__ = 'src/engine/SCons/Variables/EnumVariable.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__all__ = ['EnumVariable']
import SCons.Errors

def _validator(key, val, env, vals):
    if False:
        i = 10
        return i + 15
    if val not in vals:
        raise SCons.Errors.UserError('Invalid value for option %s: %s.  Valid values are: %s' % (key, val, vals))

def EnumVariable(key, help, default, allowed_values, map={}, ignorecase=0):
    if False:
        i = 10
        return i + 15
    "\n    The input parameters describe an option with only certain values\n    allowed. They are returned with an appropriate converter and\n    validator appended. The result is usable for input to\n    Variables.Add().\n\n    'key' and 'default' are the values to be passed on to Variables.Add().\n\n    'help' will be appended by the allowed values automatically\n\n    'allowed_values' is a list of strings, which are allowed as values\n    for this option.\n\n    The 'map'-dictionary may be used for converting the input value\n    into canonical values (e.g. for aliases).\n\n    'ignorecase' defines the behaviour of the validator:\n\n        If ignorecase == 0, the validator/converter are case-sensitive.\n        If ignorecase == 1, the validator/converter are case-insensitive.\n        If ignorecase == 2, the validator/converter is case-insensitive and the converted value will always be lower-case.\n\n    The 'validator' tests whether the value is in the list of allowed values. The 'converter' converts input values\n    according to the given 'map'-dictionary (unmapped input values are returned unchanged).\n    "
    help = '%s (%s)' % (help, '|'.join(allowed_values))
    if ignorecase >= 1:
        validator = lambda key, val, env: _validator(key, val.lower(), env, allowed_values)
    else:
        validator = lambda key, val, env: _validator(key, val, env, allowed_values)
    if ignorecase == 2:
        converter = lambda val: map.get(val.lower(), val).lower()
    elif ignorecase == 1:
        converter = lambda val: map.get(val.lower(), val)
    else:
        converter = lambda val: map.get(val, val)
    return (key, help, default, validator, converter)