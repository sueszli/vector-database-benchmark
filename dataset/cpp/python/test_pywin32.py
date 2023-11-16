#-----------------------------------------------------------------------------
# Copyright (c) 2021-2023, PyInstaller Development Team.
#
# Distributed under the terms of the GNU General Public License (version 2
# or later) with exception for distributing the bootloader.
#
# The full license is in the file COPYING.txt, distributed with this software.
#
# SPDX-License-Identifier: (GPL-2.0-or-later WITH Bootloader-exception)
#-----------------------------------------------------------------------------

import pytest

from PyInstaller.utils.tests import importorskip
from PyInstaller.utils.hooks import can_import_module


# pywin32 provides several modules that can be imported on their own, such as win32api or win32com.
# They are located in three directories, which are added to search path via pywin32.pth:
# - win32
# - win32/lib
# - pythonwin
@importorskip('win32api')  # Use win32api to check for presence of pywin32.
@pytest.mark.parametrize(
    'module',
    (
        # Binary extensions from win32
        'mmapfile',
        'odbc',
        'perfmon',
        'servicemanager',
        'timer',
        'win32api',
        'win32clipboard',
        'win32console',
        'win32cred',
        'win32crypt',
        'win32event',
        'win32evtlog',
        'win32file',
        'win32gui',
        'win32help',
        'win32inet',
        'win32job',
        'win32lz',
        'win32net',
        'win32pdh',
        'win32pipe',
        'win32print',
        'win32process',
        'win32profile',
        'win32ras',
        'win32security',
        'win32service',
        '_win32sysloader',
        'win32trace',
        'win32transaction',
        'win32ts',
        'win32wnet',
        '_winxptheme',
        # Python modules from win32/lib
        'afxres',
        'commctrl',
        'dbi',
        'mmsystem',
        'netbios',
        'ntsecuritycon',
        'pywin32_bootstrap',
        'pywin32_testutil',
        'pywintypes',
        'rasutil',
        'regcheck',
        'regutil',
        'sspicon',
        'sspi',
        'win2kras',
        'win32con',
        'win32cryptcon',
        'win32evtlogutil',
        'win32gui_struct',
        'win32inetcon',
        'win32netcon',
        'win32pdhquery',
        'win32pdhutil',
        'win32rcparser',
        'win32serviceutil',
        'win32timezone',
        'win32traceutil',
        'win32verstamp',
        'winerror',
        'winioctlcon',
        'winnt',
        'winperf',
        'winxptheme',
        # Binary extensions from pythonwin
        'dde',
        'win32uiole',
        'win32ui',
        # Python package from pythonwin
        'pywin',
        # Packages/modules in top-level directory
        'adodbapi',
        'isapi',
        'win32com',
        'win32comext',
        'pythoncom',
    )
)
@pytest.mark.parametrize('pyi_builder', ['onedir'], indirect=True)  # Run only in onedir mode.
def test_pywin32_imports(pyi_builder, module):
    if not can_import_module(module):
        pytest.skip(f"Module '{module}' cannot be imported.")

    # Basic import test
    pyi_builder.test_source(f"""
        import {module}
        """)
