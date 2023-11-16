import json
from pathlib import Path
import pytest
from redbot.pytest.downloader import *
from redbot.cogs.downloader.installable import Installable, InstallableType
from redbot.core import VersionInfo

def test_process_info_file(installable):
    if False:
        return 10
    for (k, v) in INFO_JSON.items():
        if k == 'type':
            assert installable.type == InstallableType.COG
        elif k in ('min_bot_version', 'max_bot_version'):
            assert getattr(installable, k) == VersionInfo.from_str(v)
        else:
            assert getattr(installable, k) == v

def test_process_lib_info_file(library_installable):
    if False:
        while True:
            i = 10
    for (k, v) in LIBRARY_INFO_JSON.items():
        if k == 'type':
            assert library_installable.type == InstallableType.SHARED_LIBRARY
        elif k in ('min_bot_version', 'max_bot_version'):
            assert getattr(library_installable, k) == VersionInfo.from_str(v)
        elif k == 'hidden':
            assert library_installable.hidden is True
        else:
            assert getattr(library_installable, k) == v

def test_location_is_dir(installable):
    if False:
        for i in range(10):
            print('nop')
    assert installable._location.exists()
    assert installable._location.is_dir()

def test_info_file_is_file(installable):
    if False:
        while True:
            i = 10
    assert installable._info_file.exists()
    assert installable._info_file.is_file()

def test_name(installable):
    if False:
        i = 10
        return i + 15
    assert installable.name == 'test_cog'

def test_repo_name(installable):
    if False:
        print('Hello World!')
    assert installable.repo_name == 'test_repo'

def test_serialization(installed_cog):
    if False:
        for i in range(10):
            print('nop')
    data = installed_cog.to_json()
    cog_name = data['module_name']
    assert cog_name == 'test_installed_cog'