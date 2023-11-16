from __future__ import absolute_import
import os
import sys
import errno
try:
    from lxml import etree
    have_lxml = True
except ImportError:
    have_lxml = False
    try:
        from xml.etree import cElementTree as etree
    except ImportError:
        try:
            from xml.etree import ElementTree as etree
        except ImportError:
            etree = None
from ..Compiler import Errors
from ..Compiler.StringEncoding import EncodedString

def is_valid_tag(name):
    if False:
        return 10
    "\n    Names like '.0' are used internally for arguments\n    to functions creating generator expressions,\n    however they are not identifiers.\n\n    See https://github.com/cython/cython/issues/5552\n    "
    if isinstance(name, EncodedString):
        if name.startswith('.') and name[1:].isdecimal():
            return False
    return True

class CythonDebugWriter(object):
    """
    Class to output debugging information for cygdb

    It writes debug information to cython_debug/cython_debug_info_<modulename>
    in the build directory.
    """

    def __init__(self, output_dir):
        if False:
            while True:
                i = 10
        if etree is None:
            raise Errors.NoElementTreeInstalledException()
        self.output_dir = os.path.join(output_dir or os.curdir, 'cython_debug')
        self.tb = etree.TreeBuilder()
        self.module_name = None
        self.start('cython_debug', attrs=dict(version='1.0'))

    def start(self, name, attrs=None):
        if False:
            print('Hello World!')
        if is_valid_tag(name):
            self.tb.start(name, attrs or {})

    def end(self, name):
        if False:
            print('Hello World!')
        if is_valid_tag(name):
            self.tb.end(name)

    def add_entry(self, name, **attrs):
        if False:
            while True:
                i = 10
        if is_valid_tag(name):
            self.tb.start(name, attrs)
            self.tb.end(name)

    def serialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.tb.end('Module')
        self.tb.end('cython_debug')
        xml_root_element = self.tb.close()
        try:
            os.makedirs(self.output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        et = etree.ElementTree(xml_root_element)
        kw = {}
        if have_lxml:
            kw['pretty_print'] = True
        fn = 'cython_debug_info_' + self.module_name
        et.write(os.path.join(self.output_dir, fn), encoding='UTF-8', **kw)
        interpreter_path = os.path.join(self.output_dir, 'interpreter')
        with open(interpreter_path, 'w') as f:
            f.write(sys.executable)