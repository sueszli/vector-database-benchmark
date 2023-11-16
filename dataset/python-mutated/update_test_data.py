"""
Script to automatically generate a JSON file containing time zone information.

This is done to allow "pinning" a small subset of the tzdata in the tests,
since we are testing properties of a file that may be subject to change. For
example, the behavior in the far future of any given zone is likely to change,
but "does this give the right answer for this file in 2040" is still an
important property to test.

This must be run from a computer with zoneinfo data installed.
"""
from __future__ import annotations
import base64
import functools
import json
import lzma
import pathlib
import textwrap
import typing
import zoneinfo
KEYS = ['Africa/Abidjan', 'Africa/Casablanca', 'America/Los_Angeles', 'America/Santiago', 'Asia/Tokyo', 'Australia/Sydney', 'Europe/Dublin', 'Europe/Lisbon', 'Europe/London', 'Pacific/Kiritimati', 'UTC']
TEST_DATA_LOC = pathlib.Path(__file__).parent

@functools.lru_cache(maxsize=None)
def get_zoneinfo_path() -> pathlib.Path:
    if False:
        print('Hello World!')
    'Get the first zoneinfo directory on TZPATH containing the "UTC" zone.'
    key = 'UTC'
    for path in map(pathlib.Path, zoneinfo.TZPATH):
        if (path / key).exists():
            return path
    else:
        raise OSError('Cannot find time zone data.')

def get_zoneinfo_metadata() -> typing.Dict[str, str]:
    if False:
        i = 10
        return i + 15
    path = get_zoneinfo_path()
    tzdata_zi = path / 'tzdata.zi'
    if not tzdata_zi.exists():
        raise OSError('Time zone data does not include tzdata.zi.')
    with open(tzdata_zi, 'r') as f:
        version_line = next(f)
    (_, version) = version_line.strip().rsplit(' ', 1)
    if not version[0:4].isdigit() or len(version) < 5 or (not version[4:].isalpha()):
        raise ValueError('Version string should be YYYYx, ' + 'where YYYY is the year and x is a letter; ' + f'found: {version}')
    return {'version': version}

def get_zoneinfo(key: str) -> bytes:
    if False:
        return 10
    path = get_zoneinfo_path()
    with open(path / key, 'rb') as f:
        return f.read()

def encode_compressed(data: bytes) -> typing.List[str]:
    if False:
        print('Hello World!')
    compressed_zone = lzma.compress(data)
    raw = base64.b85encode(compressed_zone)
    raw_data_str = raw.decode('utf-8')
    data_str = textwrap.wrap(raw_data_str, width=70)
    return data_str

def load_compressed_keys() -> typing.Dict[str, typing.List[str]]:
    if False:
        i = 10
        return i + 15
    output = {key: encode_compressed(get_zoneinfo(key)) for key in KEYS}
    return output

def update_test_data(fname: str='zoneinfo_data.json') -> None:
    if False:
        for i in range(10):
            print('nop')
    TEST_DATA_LOC.mkdir(exist_ok=True, parents=True)
    json_kwargs: typing.Dict[str, typing.Any] = dict(indent=2, sort_keys=True)
    compressed_keys = load_compressed_keys()
    metadata = get_zoneinfo_metadata()
    output = {'metadata': metadata, 'data': compressed_keys}
    with open(TEST_DATA_LOC / fname, 'w') as f:
        json.dump(output, f, **json_kwargs)
if __name__ == '__main__':
    update_test_data()