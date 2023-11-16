import pytest
from PyInstaller.utils.tests import importorskip
from PyInstaller.utils.hooks import can_import_module

@importorskip('win32api')
@pytest.mark.parametrize('module', ('mmapfile', 'odbc', 'perfmon', 'servicemanager', 'timer', 'win32api', 'win32clipboard', 'win32console', 'win32cred', 'win32crypt', 'win32event', 'win32evtlog', 'win32file', 'win32gui', 'win32help', 'win32inet', 'win32job', 'win32lz', 'win32net', 'win32pdh', 'win32pipe', 'win32print', 'win32process', 'win32profile', 'win32ras', 'win32security', 'win32service', '_win32sysloader', 'win32trace', 'win32transaction', 'win32ts', 'win32wnet', '_winxptheme', 'afxres', 'commctrl', 'dbi', 'mmsystem', 'netbios', 'ntsecuritycon', 'pywin32_bootstrap', 'pywin32_testutil', 'pywintypes', 'rasutil', 'regcheck', 'regutil', 'sspicon', 'sspi', 'win2kras', 'win32con', 'win32cryptcon', 'win32evtlogutil', 'win32gui_struct', 'win32inetcon', 'win32netcon', 'win32pdhquery', 'win32pdhutil', 'win32rcparser', 'win32serviceutil', 'win32timezone', 'win32traceutil', 'win32verstamp', 'winerror', 'winioctlcon', 'winnt', 'winperf', 'winxptheme', 'dde', 'win32uiole', 'win32ui', 'pywin', 'adodbapi', 'isapi', 'win32com', 'win32comext', 'pythoncom'))
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)
def test_pywin32_imports(pyi_builder, module):
    if False:
        print('Hello World!')
    if not can_import_module(module):
        pytest.skip(f"Module '{module}' cannot be imported.")
    pyi_builder.test_source(f'\n        import {module}\n        ')