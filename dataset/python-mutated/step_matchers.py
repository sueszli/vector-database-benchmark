"""
Official API for step writers that want to use step-matchers.
"""
from __future__ import absolute_import, print_function
import warnings
from behave import matchers as _step_matchers

def register_type(**kwargs):
    if False:
        return 10
    _step_matchers.register_type(**kwargs)

def use_default_step_matcher(name=None):
    if False:
        print('Hello World!')
    return _step_matchers.use_default_step_matcher(name=name)

def use_step_matcher(name):
    if False:
        while True:
            i = 10
    return _step_matchers.use_step_matcher(name)

def step_matcher(name):
    if False:
        return 10
    'DEPRECATED, use :func:`use_step_matcher()` instead.'
    warnings.warn("deprecated: Use 'use_step_matcher()' instead", DeprecationWarning, stacklevel=2)
    return use_step_matcher(name)
register_type.__doc__ = _step_matchers.register_type.__doc__
use_step_matcher.__doc__ = _step_matchers.use_step_matcher.__doc__
use_default_step_matcher.__doc__ = _step_matchers.use_default_step_matcher.__doc__