from __future__ import annotations
import contextlib
import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator
from typing import cast
from pendulum.tz.exceptions import InvalidTimezone
from pendulum.tz.timezone import UTC
from pendulum.tz.timezone import FixedTimezone
from pendulum.tz.timezone import Timezone
if sys.platform == 'win32':
    import winreg
_mock_local_timezone = None
_local_timezone = None

def get_local_timezone() -> Timezone | FixedTimezone:
    if False:
        print('Hello World!')
    global _local_timezone
    if _mock_local_timezone is not None:
        return _mock_local_timezone
    if _local_timezone is None:
        tz = _get_system_timezone()
        _local_timezone = tz
    return _local_timezone

def set_local_timezone(mock: str | Timezone | None=None) -> None:
    if False:
        print('Hello World!')
    global _mock_local_timezone
    _mock_local_timezone = mock

@contextmanager
def test_local_timezone(mock: Timezone) -> Iterator[None]:
    if False:
        print('Hello World!')
    set_local_timezone(mock)
    yield
    set_local_timezone()

def _get_system_timezone() -> Timezone:
    if False:
        for i in range(10):
            print('nop')
    if sys.platform == 'win32':
        return _get_windows_timezone()
    elif 'darwin' in sys.platform:
        return _get_darwin_timezone()
    return _get_unix_timezone()
if sys.platform == 'win32':

    def _get_windows_timezone() -> Timezone:
        if False:
            while True:
                i = 10
        from pendulum.tz.data.windows import windows_timezones
        handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        tz_local_key_name = 'SYSTEM\\CurrentControlSet\\Control\\TimeZoneInformation'
        localtz = winreg.OpenKey(handle, tz_local_key_name)
        timezone_info = {}
        size = winreg.QueryInfoKey(localtz)[1]
        for i in range(size):
            data = winreg.EnumValue(localtz, i)
            timezone_info[data[0]] = data[1]
        localtz.Close()
        if 'TimeZoneKeyName' in timezone_info:
            tzkeyname = timezone_info['TimeZoneKeyName'].split('\x00', 1)[0]
        else:
            tzwin = timezone_info['StandardName']
            tz_key_name = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Time Zones'
            tzkey = winreg.OpenKey(handle, tz_key_name)
            tzkeyname = None
            for i in range(winreg.QueryInfoKey(tzkey)[0]):
                subkey = winreg.EnumKey(tzkey, i)
                sub = winreg.OpenKey(tzkey, subkey)
                info = {}
                size = winreg.QueryInfoKey(sub)[1]
                for i in range(size):
                    data = winreg.EnumValue(sub, i)
                    info[data[0]] = data[1]
                sub.Close()
                with contextlib.suppress(KeyError):
                    if info['Std'] == tzwin:
                        tzkeyname = subkey
                        break
            tzkey.Close()
            handle.Close()
        if tzkeyname is None:
            raise LookupError('Can not find Windows timezone configuration')
        timezone = windows_timezones.get(tzkeyname)
        if timezone is None:
            timezone = windows_timezones.get(tzkeyname + ' Standard Time')
        if timezone is None:
            raise LookupError('Unable to find timezone ' + tzkeyname)
        return Timezone(timezone)
else:

    def _get_windows_timezone() -> Timezone:
        if False:
            return 10
        raise NotImplementedError

def _get_darwin_timezone() -> Timezone:
    if False:
        print('Hello World!')
    link = os.readlink('/etc/localtime')
    tzname = link[link.rfind('zoneinfo/') + 9:]
    return Timezone(tzname)

def _get_unix_timezone(_root: str='/') -> Timezone:
    if False:
        while True:
            i = 10
    tzenv = os.environ.get('TZ')
    if tzenv:
        with contextlib.suppress(ValueError):
            return _tz_from_env(tzenv)
    tzpath = os.path.join(_root, 'etc/timezone')
    if os.path.isfile(tzpath):
        with open(tzpath, 'rb') as tzfile:
            tzfile_data = tzfile.read()
            if tzfile_data[:5] != b'TZif2':
                etctz = tzfile_data.strip().decode()
                if ' ' in etctz:
                    (etctz, dummy) = etctz.split(' ', 1)
                if '#' in etctz:
                    (etctz, dummy) = etctz.split('#', 1)
                return Timezone(etctz.replace(' ', '_'))
    zone_re = re.compile('\\s*ZONE\\s*=\\s*"')
    timezone_re = re.compile('\\s*TIMEZONE\\s*=\\s*"')
    end_re = re.compile('"')
    for filename in ('etc/sysconfig/clock', 'etc/conf.d/clock'):
        tzpath = os.path.join(_root, filename)
        if not os.path.isfile(tzpath):
            continue
        with open(tzpath) as tzfile:
            data = tzfile.readlines()
        for line in data:
            match = zone_re.match(line)
            if match is None:
                match = timezone_re.match(line)
            if match is not None:
                line = line[match.end():]
                etctz = line[:cast(re.Match, end_re.search(line)).start()]
                parts = list(reversed(etctz.replace(' ', '_').split(os.path.sep)))
                tzpath_parts: list[str] = []
                while parts:
                    tzpath_parts.insert(0, parts.pop(0))
                    with contextlib.suppress(InvalidTimezone):
                        return Timezone(os.path.join(*tzpath_parts))
    tzpath = os.path.join(_root, 'etc', 'localtime')
    if os.path.isfile(tzpath) and os.path.islink(tzpath):
        parts = list(reversed(os.path.realpath(tzpath).replace(' ', '_').split(os.path.sep)))
        tzpath_parts: list[str] = []
        while parts:
            tzpath_parts.insert(0, parts.pop(0))
            with contextlib.suppress(InvalidTimezone):
                return Timezone(os.path.join(*tzpath_parts))
    for filename in ('etc/localtime', 'usr/local/etc/localtime'):
        tzpath = os.path.join(_root, filename)
        if not os.path.isfile(tzpath):
            continue
        with open(tzpath, 'rb') as f:
            return Timezone.from_file(f)
    warnings.warn('Unable not find any timezone configuration, defaulting to UTC.', stacklevel=1)
    return UTC

def _tz_from_env(tzenv: str) -> Timezone:
    if False:
        return 10
    if tzenv[0] == ':':
        tzenv = tzenv[1:]
    if os.path.isfile(tzenv):
        with open(tzenv, 'rb') as f:
            return Timezone.from_file(f)
    try:
        return Timezone(tzenv)
    except ValueError:
        raise