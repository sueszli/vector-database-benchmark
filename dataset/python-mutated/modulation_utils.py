"""
Miscellaneous utilities for managing mods and demods, as well as other items
useful in dealing with generalized handling of different modulations and demods.
"""
import inspect
_type_1_modulators = {}

def type_1_mods():
    if False:
        print('Hello World!')
    return _type_1_modulators

def add_type_1_mod(name, mod_class):
    if False:
        for i in range(10):
            print('nop')
    _type_1_modulators[name] = mod_class
_type_1_demodulators = {}

def type_1_demods():
    if False:
        print('Hello World!')
    return _type_1_demodulators

def add_type_1_demod(name, demod_class):
    if False:
        while True:
            i = 10
    _type_1_demodulators[name] = demod_class
_type_1_constellations = {}

def type_1_constellations():
    if False:
        return 10
    return _type_1_constellations

def add_type_1_constellation(name, constellation):
    if False:
        for i in range(10):
            print('nop')
    _type_1_constellations[name] = constellation

def extract_kwargs_from_options(function, excluded_args, options):
    if False:
        while True:
            i = 10
    "\n    Given a function, a list of excluded arguments and the result of\n    parsing command line options, create a dictionary of key word\n    arguments suitable for passing to the function.  The dictionary\n    will be populated with key/value pairs where the keys are those\n    that are common to the function's argument list (minus the\n    excluded_args) and the attributes in options.  The values are the\n    corresponding values from options unless that value is None.\n    In that case, the corresponding dictionary entry is not populated.\n\n    (This allows different modulations that have the same parameter\n    names, but different default values to coexist.  The downside is\n    that --help in the option parser will list the default as None,\n    but in that case the default provided in the __init__ argument\n    list will be used since there is no kwargs entry.)\n\n    Args:\n        function: the function whose parameter list will be examined\n        excluded_args: function arguments that are NOT to be added to the dictionary (sequence of strings)\n        options: result of command argument parsing (optparse.Values)\n    "
    spec = inspect.getfullargspec(function)
    d = {}
    for kw in [a for a in spec.args if a not in excluded_args]:
        if hasattr(options, kw):
            if getattr(options, kw) is not None:
                d[kw] = getattr(options, kw)
    return d

def extract_kwargs_from_options_for_class(cls, options):
    if False:
        while True:
            i = 10
    '\n    Given command line options, create dictionary suitable for passing to __init__\n    '
    d = extract_kwargs_from_options(cls.__init__, ('self',), options)
    for base in cls.__bases__:
        if hasattr(base, 'extract_kwargs_from_options'):
            d.update(base.extract_kwargs_from_options(options))
    return d