"""
This compat module wraps os.path to forbid some functions.

isort:skip_file
"""
from __future__ import absolute_import
from os.path import *
import os.path as std_os_path
import sys as std_sys
ourselves = std_sys.modules[__name__]
for attribute in dir(std_os_path):
    if not hasattr(ourselves, attribute):
        setattr(ourselves, attribute, getattr(std_os_path, attribute))
del ourselves, std_os_path, std_sys

def realpath(*unused_args, **unused_kwargs):
    if False:
        while True:
            i = 10
    'Method os.path.realpath() is forbidden'
    raise RuntimeError('Usage of os.path.realpath() is forbidden. Use certbot.compat.filesystem.realpath() instead.')