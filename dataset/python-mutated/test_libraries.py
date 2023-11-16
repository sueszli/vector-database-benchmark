import os
import pytest
import py
from PyInstaller.compat import is_win, is_linux
from PyInstaller.utils.tests import importorskip, xfail, skipif, requires
from PyInstaller.utils.hooks import can_import_module
_MODULES_DIR = py.path.local(os.path.abspath(__file__)).dirpath('modules')
_DATA_DIR = py.path.local(os.path.abspath(__file__)).dirpath('data')

@importorskip('gevent')
def test_gevent(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source('\n        import gevent\n        gevent.spawn(lambda: x)\n        ')

@importorskip('gevent')
def test_gevent_monkey(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source('\n        from gevent.monkey import patch_all\n        patch_all()\n        ')

@pytest.mark.skipif(not can_import_module('tkinter'), reason='tkinter cannot be imported.')
def test_tkinter(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_script('pyi_lib_tkinter.py')

def test_pkg_resource_res_string(pyi_builder, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    datas = os.pathsep.join((str(_MODULES_DIR.join('pkg3', 'sample-data.txt')), 'pkg3'))
    pyi_builder.test_script('pkg_resource_res_string.py', pyi_args=['--add-data', datas])

def test_pkgutil_get_data(pyi_builder, monkeypatch):
    if False:
        i = 10
        return i + 15
    datas = os.pathsep.join((str(_MODULES_DIR.join('pkg3', 'sample-data.txt')), 'pkg3'))
    pyi_builder.test_script('pkgutil_get_data.py', pyi_args=['--add-data', datas])

@xfail(reason='Our import mechanism returns the wrong loader-class for __main__.')
def test_pkgutil_get_data__main__(pyi_builder, monkeypatch):
    if False:
        i = 10
        return i + 15
    datas = os.pathsep.join((str(_MODULES_DIR.join('pkg3', 'sample-data.txt')), 'pkg3'))
    pyi_builder.test_script('pkgutil_get_data__main__.py', pyi_args=['--add-data', datas])

@importorskip('sphinx')
def test_sphinx(tmpdir, pyi_builder, data_dir):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_script('pyi_lib_sphinx.py')

@importorskip('pygments')
def test_pygments(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source('\n        # This sample code is taken from http://pygments.org/docs/quickstart/.\n        from pygments import highlight\n        from pygments.lexers import PythonLexer\n        from pygments.formatters import HtmlFormatter\n\n        code = \'print "Hello World"\'\n        print(highlight(code, PythonLexer(), HtmlFormatter()))\n        ')

@requires('zope.interface')
def test_zope_interface(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source("\n        # Package 'zope' does not contain __init__.py file.\n        # Just importing 'zope.interface' is sufficient.\n        import zope.interface\n        ")

@importorskip('idlelib')
@pytest.mark.skipif(not can_import_module('tkinter'), reason='tkinter cannot be imported.')
def test_idlelib(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source('\n        # This file depends on loading some icons, located based on __file__.\n        import idlelib.tree\n        ')

@importorskip('keyring')
@skipif(is_linux, reason='SecretStorage backend on linux requires active D-BUS session and initialized keyring, and may need to unlock the keyring via UI prompt.')
def test_keyring(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source('\n        import keyring\n        keyring.get_password("test", "test")\n        ')

@importorskip('numpy')
def test_numpy(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source("\n        import numpy\n        from numpy.core.numeric import dot\n        print('dot(3, 4):', dot(3, 4))\n        ")

@importorskip('pytz')
def test_pytz(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source("\n        import pytz\n        pytz.timezone('US/Eastern')\n        ")

@importorskip('requests')
def test_requests(tmpdir, pyi_builder, data_dir, monkeypatch):
    if False:
        print('Hello World!')
    datas = os.pathsep.join((str(data_dir.join('*')), os.curdir))
    pyi_builder.test_script('pyi_lib_requests.py', pyi_args=['--add-data', datas])

@importorskip('urllib3.packages.six')
def test_urllib3_six(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source('\n        import urllib3.connectionpool\n        import types\n        assert isinstance(urllib3.connectionpool.queue, types.ModuleType)\n        ')

@importorskip('sqlite3')
def test_sqlite3(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source("\n        # PyInstaller did not included module 'sqlite3.dump'.\n        import sqlite3\n        conn = sqlite3.connect(':memory:')\n        csr = conn.cursor()\n        csr.execute('CREATE TABLE Example (id)')\n        for line in conn.iterdump():\n             print(line)\n        ")

@requires('scapy >= 2.0')
def test_scapy(pyi_builder):
    if False:
        print('Hello World!')
    pyi_builder.test_source('\n        # Test-cases taken from issue #834\n        import scapy.all\n        scapy.all.IP\n\n        from scapy.all import IP\n\n        # Test-case taken from issue #202.\n        from scapy.all import *\n        DHCP  # scapy.layers.dhcp.DHCP\n        BOOTP  # scapy.layers.dhcp.BOOTP\n        DNS  # scapy.layers.dns.DNS\n        ICMP  # scapy.layers.inet.ICMP\n        ')

@requires('scapy >= 2.0')
def test_scapy2(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source('\n        # Test the hook to scapy.layers.all\n        from scapy.layers.all import DHCP\n        ')

@requires('scapy >= 2.0')
def test_scapy3(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source("\n        # Test whether\n        # a) scapy packet layers are not included if neither scapy.all nor scapy.layers.all are imported\n        # b) packages are included if imported explicitly\n\n        NAME = 'hook-scapy.layers.all'\n        layer_inet = 'scapy.layers.inet'\n\n        def testit():\n            try:\n                __import__(layer_inet)\n                raise SystemExit('Self-test of hook %s failed: package module found'\n                                 % NAME)\n            except ImportError, e:\n                if not e.args[0].endswith(' inet'):\n                    raise SystemExit('Self-test of hook %s failed: package module found and has import errors: %r'\n                                     % (NAME, e))\n\n        import scapy\n        testit()\n        import scapy.layers\n        testit()\n        # Explicitly import a single layer module. Note: This module MUST NOT import inet (neither directly nor\n        # indirectly), otherwise the test above fails.\n        import scapy.layers.ir\n        ")

@importorskip('sqlalchemy')
def test_sqlalchemy(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source('\n        # The hook behaviour is to include with sqlalchemy all installed database backends.\n        import sqlalchemy\n        # This import was known to fail with sqlalchemy 0.9.1\n        import sqlalchemy.ext.declarative\n        ')

@importorskip('twisted')
def test_twisted(pyi_builder):
    if False:
        print('Hello World!')
    pyi_builder.test_source("\n        # Twisted is an event-driven networking engine.\n        #\n        # The 'reactor' is object that starts the eventloop.\n        # There are different types of platform specific reactors.\n        # Platform specific reactor is wrapped into twisted.internet.reactor module.\n        from twisted.internet import reactor\n        # Applications importing module twisted.internet.reactor might fail with error like:\n        #\n        #     AttributeError: 'module' object has no attribute 'listenTCP'\n        #\n        # Ensure default reactor was loaded - it has method 'listenTCP' to start server.\n        if not hasattr(reactor, 'listenTCP'):\n            raise SystemExit('Twisted reactor not properly initialized.')\n        ")

@importorskip('pyexcelerate')
def test_pyexcelerate(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source('\n        # Requires PyExcelerate 0.6.1 or higher\n        # Tested on Windows 7 x64 SP1 with CPython 2.7.6\n        import pyexcelerate\n        ')

@importorskip('usb')
@pytest.mark.skipif(is_linux, reason='libusb_exit segfaults on some linuxes')
def test_usb(pyi_builder):
    if False:
        return 10
    try:
        import usb
        usb.core.find()
    except (ImportError, usb.core.NoBackendError):
        pytest.skip('USB backnd not found.')
    pyi_builder.test_source('\n        import usb.core\n        # NoBackendError fails the test if no backends are found.\n        usb.core.find()\n        ')

@importorskip('zeep')
def test_zeep(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source('\n        # Test the hook to zeep\n        from zeep import utils\n        utils.get_version()\n        ')

@importorskip('PIL')
def test_pil_img_conversion(pyi_builder):
    if False:
        print('Hello World!')
    datas = os.pathsep.join((str(_DATA_DIR.join('PIL_images')), '.'))
    pyi_builder.test_script('pyi_lib_PIL_img_conversion.py', pyi_args=['--add-data', datas, '--console'])

@requires('pillow >= 1.1.6')
@importorskip('PyQt5')
def test_pil_PyQt5(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source('\n        import PyQt5\n        import PIL\n        import PIL.ImageQt\n        ')

@importorskip('PIL')
def test_pil_plugins(pyi_builder):
    if False:
        print('Hello World!')
    pyi_builder.test_source("\n        # Verify packaging of PIL.Image.\n        from PIL.Image import frombytes\n        print(frombytes)\n\n        # PIL import hook should bundle all available PIL plugins. Verify that plugins are collected.\n        from PIL import Image\n        Image.init()\n        MIN_PLUG_COUNT = 7  # Without all plugins the count is usually 6.\n        plugins = list(Image.SAVE.keys())\n        plugins.sort()\n        if len(plugins) < MIN_PLUG_COUNT:\n            raise SystemExit('No PIL image plugins were collected!')\n        else:\n            print('PIL supported image formats: %s' % plugins)\n        ")

@importorskip('pandas')
def test_pandas_extension(pyi_builder):
    if False:
        while True:
            i = 10
    pyi_builder.test_source('\n        from pandas._libs.lib import is_float\n        assert is_float(1) == 0\n        ')

@importorskip('pandas')
@importorskip('jinja2')
def test_pandas_io_formats_style(pyi_builder):
    if False:
        return 10
    pyi_builder.test_source('\n        import pandas.io.formats.style\n        ')

@importorskip('pandas')
@importorskip('matplotlib')
def test_pandas_plotting_matplotlib(pyi_builder):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source("\n        import matplotlib as mpl\n        import pandas as pd\n\n        mpl.use('Agg')  # Use headless Agg backend to avoid dependency on display server.\n\n        series = pd.Series([0, 1, 2, 3], [0, 1, 2, 3])\n        series.plot()\n        ")

@importorskip('win32ctypes')
@pytest.mark.skipif(not is_win, reason='pywin32-ctypes is supported only on Windows')
@pytest.mark.parametrize('submodule', ['win32api', 'win32cred', 'pywintypes'])
def test_pywin32ctypes(pyi_builder, submodule):
    if False:
        i = 10
        return i + 15
    pyi_builder.test_source('\n        from win32ctypes.pywin32 import {0}\n        '.format(submodule))

@importorskip('setuptools')
def test_setuptools(pyi_builder):
    if False:
        for i in range(10):
            print('nop')
    pyi_builder.test_source('\n        import setuptools\n        ')