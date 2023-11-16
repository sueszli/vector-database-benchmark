from __future__ import annotations
import codecs
import json
import locale
import os
import platform
import struct
import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas._typing import JSONSerializable
from pandas.compat._optional import VERSIONS, get_version, import_optional_dependency

def _get_commit_hash() -> str | None:
    if False:
        print('Hello World!')
    '\n    Use vendored versioneer code to get git hash, which handles\n    git worktree correctly.\n    '
    try:
        from pandas._version_meson import __git_version__
        return __git_version__
    except ImportError:
        from pandas._version import get_versions
        versions = get_versions()
        return versions['full-revisionid']

def _get_sys_info() -> dict[str, JSONSerializable]:
    if False:
        print('Hello World!')
    '\n    Returns system information as a JSON serializable dictionary.\n    '
    uname_result = platform.uname()
    (language_code, encoding) = locale.getlocale()
    return {'commit': _get_commit_hash(), 'python': '.'.join([str(i) for i in sys.version_info]), 'python-bits': struct.calcsize('P') * 8, 'OS': uname_result.system, 'OS-release': uname_result.release, 'Version': uname_result.version, 'machine': uname_result.machine, 'processor': uname_result.processor, 'byteorder': sys.byteorder, 'LC_ALL': os.environ.get('LC_ALL'), 'LANG': os.environ.get('LANG'), 'LOCALE': {'language-code': language_code, 'encoding': encoding}}

def _get_dependency_info() -> dict[str, JSONSerializable]:
    if False:
        while True:
            i = 10
    '\n    Returns dependency information as a JSON serializable dictionary.\n    '
    deps = ['pandas', 'numpy', 'pytz', 'dateutil', 'setuptools', 'pip', 'Cython', 'pytest', 'hypothesis', 'sphinx', 'blosc', 'feather', 'xlsxwriter', 'lxml.etree', 'html5lib', 'pymysql', 'psycopg2', 'jinja2', 'IPython', 'pandas_datareader']
    deps.extend(list(VERSIONS))
    result: dict[str, JSONSerializable] = {}
    for modname in deps:
        mod = import_optional_dependency(modname, errors='ignore')
        result[modname] = get_version(mod) if mod else None
    return result

def show_versions(as_json: str | bool=False) -> None:
    if False:
        print('Hello World!')
    '\n    Provide useful information, important for bug reports.\n\n    It comprises info about hosting operation system, pandas version,\n    and versions of other installed relative packages.\n\n    Parameters\n    ----------\n    as_json : str or bool, default False\n        * If False, outputs info in a human readable form to the console.\n        * If str, it will be considered as a path to a file.\n          Info will be written to that file in JSON format.\n        * If True, outputs info in JSON format to the console.\n\n    Examples\n    --------\n    >>> pd.show_versions()  # doctest: +SKIP\n    Your output may look something like this:\n    INSTALLED VERSIONS\n    ------------------\n    commit           : 37ea63d540fd27274cad6585082c91b1283f963d\n    python           : 3.10.6.final.0\n    python-bits      : 64\n    OS               : Linux\n    OS-release       : 5.10.102.1-microsoft-standard-WSL2\n    Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022\n    machine          : x86_64\n    processor        : x86_64\n    byteorder        : little\n    LC_ALL           : None\n    LANG             : en_GB.UTF-8\n    LOCALE           : en_GB.UTF-8\n    pandas           : 2.0.1\n    numpy            : 1.24.3\n    ...\n    '
    sys_info = _get_sys_info()
    deps = _get_dependency_info()
    if as_json:
        j = {'system': sys_info, 'dependencies': deps}
        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        else:
            assert isinstance(as_json, str)
            with codecs.open(as_json, 'wb', encoding='utf8') as f:
                json.dump(j, f, indent=2)
    else:
        assert isinstance(sys_info['LOCALE'], dict)
        language_code = sys_info['LOCALE']['language-code']
        encoding = sys_info['LOCALE']['encoding']
        sys_info['LOCALE'] = f'{language_code}.{encoding}'
        maxlen = max((len(x) for x in deps))
        print('\nINSTALLED VERSIONS')
        print('------------------')
        for (k, v) in sys_info.items():
            print(f'{k:<{maxlen}}: {v}')
        print('')
        for (k, v) in deps.items():
            print(f'{k:<{maxlen}}: {v}')