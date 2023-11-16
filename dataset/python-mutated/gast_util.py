"""Gast compatibility library. Supports 0.2.2 and 0.3.2."""
import functools
import gast
from distutils.version import LooseVersion

def get_gast_version():
    if False:
        print('Hello World!')
    'Gast exports `__version__` from 0.5.3 onwards, we need to look it up in a different way.'
    if hasattr(gast, '__version__'):
        return gast.__version__
    try:
        import pkg_resources
        return pkg_resources.get_distribution('gast').version
    except pkg_resources.DistributionNotFound:
        if hasattr(gast, 'Str'):
            return '0.2'
        else:
            try:
                gast.Assign(None, None, None)
            except AssertionError as e:
                if 'Bad argument number for Assign: 3, expecting 2' in str(e):
                    return '0.4'
            return '0.5'

def is_constant(node):
    if False:
        i = 10
        return i + 15
    'Tests whether node represents a Python constant.'
    return isinstance(node, gast.Constant)

def is_literal(node):
    if False:
        i = 10
        return i + 15
    'Tests whether node represents a Python literal.'
    if is_constant(node):
        return True
    if isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']:
        return True
    return False

def is_ellipsis(node):
    if False:
        for i in range(10):
            print('nop')
    'Tests whether node represents a Python ellipsis.'
    return isinstance(node, gast.Constant) and node.value == Ellipsis

def _compat_assign_gast_4(targets, value, type_comment):
    if False:
        return 10
    'Wraps around gast.Assign to use same function signature across versions.'
    return gast.Assign(targets=targets, value=value)

def _compat_assign_gast_5(targets, value, type_comment):
    if False:
        while True:
            i = 10
    'Wraps around gast.Assign to use same function signature across versions.'
    return gast.Assign(targets=targets, value=value, type_comment=type_comment)
if get_gast_version() < LooseVersion('0.5'):
    compat_assign = _compat_assign_gast_4
else:
    compat_assign = _compat_assign_gast_5
Module = functools.partial(gast.Module, type_ignores=None)
Name = functools.partial(gast.Name, type_comment=None)
Str = functools.partial(gast.Constant, kind=None)