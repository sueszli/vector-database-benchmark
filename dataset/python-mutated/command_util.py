"""
Provides some command utility functions.

TODO:
  matcher that ignores empty lines and whitespace and has contains comparison
"""
from __future__ import absolute_import, print_function
from behave4cmd0 import pathutil
from behave4cmd0.__setup import TOP, TOPA
import os.path
import sys
import shutil
import time
import tempfile
from fnmatch import fnmatch
WORKDIR = os.path.join(TOP, '__WORKDIR__')

def workdir_save_coverage_files(workdir, destdir=None):
    if False:
        while True:
            i = 10
    assert os.path.isdir(workdir)
    if not destdir:
        destdir = TOPA
    if os.path.abspath(workdir) == os.path.abspath(destdir):
        return
    for fname in os.listdir(workdir):
        if fnmatch(fname, '.coverage.*'):
            sourcename = os.path.join(workdir, fname)
            shutil.move(sourcename, destdir)

def ensure_context_attribute_exists(context, name, default_value=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure a behave resource exists as attribute in the behave context.\n    If this is not the case, the attribute is created by using the default_value.\n    '
    if not hasattr(context, name):
        setattr(context, name, default_value)

def ensure_workdir_exists(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensures that the work directory exists.\n    In addition, the location of the workdir is stored as attribute in\n    the context object.\n    '
    ensure_context_attribute_exists(context, 'workdir', None)
    if not context.workdir:
        context.workdir = os.path.abspath(WORKDIR)
    pathutil.ensure_directory_exists(context.workdir)

def ensure_workdir_not_exists(context):
    if False:
        i = 10
        return i + 15
    'Ensures that the work directory does not exist.'
    ensure_context_attribute_exists(context, 'workdir', None)
    if context.workdir:
        orig_dirname = real_dirname = context.workdir
        context.workdir = None
        if os.path.exists(real_dirname):
            renamed_dirname = tempfile.mktemp(prefix=os.path.basename(real_dirname), suffix='_DEAD', dir=os.path.dirname(real_dirname) or '.')
            os.rename(real_dirname, renamed_dirname)
            real_dirname = renamed_dirname
        max_iterations = 2
        if sys.platform.startswith('win'):
            max_iterations = 15
        for iteration in range(max_iterations):
            if not os.path.exists(real_dirname):
                if iteration > 1:
                    print('REMOVE-WORKDIR after %s iterations' % (iteration + 1))
                break
            shutil.rmtree(real_dirname, ignore_errors=True)
            time.sleep(0.5)
        assert not os.path.isdir(real_dirname), 'ENSURE not-isa dir: %s' % real_dirname
        assert not os.path.exists(real_dirname), 'ENSURE dir not-exists: %s' % real_dirname
        assert not os.path.isdir(orig_dirname), 'ENSURE not-isa dir: %s' % orig_dirname