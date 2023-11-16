"""
Finds locations of Windows command-line development tools.
"""
import os
import platform
import glob
import re
import subprocess
import sys
import textwrap

class WindowsVS:
    """
    Windows only. Finds locations of Visual Studio command-line tools. Assumes
    VS2019-style paths.

    Members and example values::

        .year:      2019
        .grade:     Community
        .version:   14.28.29910
        .directory: C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community
        .vcvars:    C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community\\VC\\Auxiliary\\Build\x0bcvars64.bat
        .cl:        C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community\\VC\\Tools\\MSVC\x0c.28.29910\x08in\\Hostx64d\\cl.exe
        .link:      C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community\\VC\\Tools\\MSVC\x0c.28.29910\x08in\\Hostx64d\\link.exe
        .csc:       C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community\\MSBuild\\Current\\Bin\\Roslyn\\csc.exe
        .devenv:    C:\\Program Files (x86)\\Microsoft Visual Studio\x819\\Community\\Common7\\IDE\\devenv.com

    `.csc` is C# compiler; will be None if not found.
    """

    def __init__(self, year=None, grade=None, version=None, cpu=None, verbose=False):
        if False:
            return 10
        '\n        Args:\n            year:\n                None or, for example, `2019`. If None we use environment\n                variable WDEV_VS_YEAR if set.\n            grade:\n                None or, for example, one of:\n\n                * `Community`\n                * `Professional`\n                * `Enterprise`\n\n                If None we use environment variable WDEV_VS_GRADE if set.\n            version:\n                None or, for example: `14.28.29910`. If None we use environment\n                variable WDEV_VS_VERSION if set.\n            cpu:\n                None or a `WindowsCpu` instance.\n        '

        def default(value, name):
            if False:
                i = 10
                return i + 15
            if value is None:
                name2 = f'WDEV_VS_{name.upper()}'
                value = os.environ.get(name2)
                if value is not None:
                    _log(f'Setting {name} from environment variable {name2}: {value!r}')
            return value
        try:
            year = default(year, 'year')
            grade = default(grade, 'grade')
            version = default(version, 'version')
            if not cpu:
                cpu = WindowsCpu()
            pattern = f"C:\\Program Files*\\Microsoft Visual Studio\\{(year if year else '2*')}\\{(grade if grade else '*')}"
            directories = glob.glob(pattern)
            if verbose:
                _log(f'Matches for: pattern={pattern!r}')
                _log(f'directories={directories!r}')
            assert directories, f'No match found for: {pattern}'
            directories.sort()
            directory = directories[-1]
            devenv = f'{directory}\\Common7\\IDE\\devenv.com'
            assert os.path.isfile(devenv), f'Does not exist: {devenv}'
            regex = f'^C:\\\\Program Files.*\\\\Microsoft Visual Studio\\\\([^\\\\]+)\\\\([^\\\\]+)'
            m = re.match(regex, directory)
            assert m, f'No match: regex={regex!r} directory={directory!r}'
            year2 = m.group(1)
            grade2 = m.group(2)
            if year:
                assert year2 == year
            else:
                year = year2
            if grade:
                assert grade2 == grade
            else:
                grade = grade2
            vcvars = f'{directory}\\VC\\Auxiliary\\Build\\vcvars{cpu.bits}.bat'
            assert os.path.isfile(vcvars), f'No match for: {vcvars}'
            cl_pattern = f"{directory}\\VC\\Tools\\MSVC\\{(version if version else '*')}\\bin\\Host{cpu.windows_name}\\{cpu.windows_name}\\cl.exe"
            cl_s = glob.glob(cl_pattern)
            assert cl_s, f'No match for: {cl_pattern}'
            cl_s.sort()
            cl = cl_s[-1]
            m = re.search(f'\\\\VC\\\\Tools\\\\MSVC\\\\([^\\\\]+)\\\\bin\\\\Host{cpu.windows_name}\\\\{cpu.windows_name}\\\\cl.exe$', cl)
            assert m
            version2 = m.group(1)
            if version:
                assert version2 == version
            else:
                version = version2
            assert version
            link_pattern = f'{directory}\\VC\\Tools\\MSVC\\{version}\\bin\\Host{cpu.windows_name}\\{cpu.windows_name}\\link.exe'
            link_s = glob.glob(link_pattern)
            assert link_s, f'No match for: {link_pattern}'
            link_s.sort()
            link = link_s[-1]
            csc = None
            for (dirpath, dirnames, filenames) in os.walk(directory):
                for filename in filenames:
                    if filename == 'csc.exe':
                        csc = os.path.join(dirpath, filename)
            self.cl = cl
            self.devenv = devenv
            self.directory = directory
            self.grade = grade
            self.link = link
            self.csc = csc
            self.vcvars = vcvars
            self.version = version
            self.year = year
        except Exception as e:
            raise Exception(f'Unable to find Visual Studio') from e

    def description_ml(self, indent=''):
        if False:
            while True:
                i = 10
        '\n        Return multiline description of `self`.\n        '
        ret = textwrap.dedent(f'\n                year:         {self.year}\n                grade:        {self.grade}\n                version:      {self.version}\n                directory:    {self.directory}\n                vcvars:       {self.vcvars}\n                cl:           {self.cl}\n                link:         {self.link}\n                csc:          {self.csc}\n                devenv:       {self.devenv}\n                ')
        return textwrap.indent(ret, indent)

    def __str__(self):
        if False:
            while True:
                i = 10
        return ' '.join(self._description())

class WindowsCpu:
    """
    For Windows only. Paths and names that depend on cpu.

    Members:
        .bits
            32 or 64.
        .windows_subdir
            Empty string or `x64/`.
        .windows_name
            `x86` or `x64`.
        .windows_config
            `x64` or `Win32`, e.g. for use in `/Build Release|x64`.
        .windows_suffix
            `64` or empty string.
    """

    def __init__(self, name=None):
        if False:
            i = 10
            return i + 15
        if not name:
            name = _cpu_name()
        self.name = name
        if name == 'x32':
            self.bits = 32
            self.windows_subdir = ''
            self.windows_name = 'x86'
            self.windows_config = 'Win32'
            self.windows_suffix = ''
        elif name == 'x64':
            self.bits = 64
            self.windows_subdir = 'x64/'
            self.windows_name = 'x64'
            self.windows_config = 'x64'
            self.windows_suffix = '64'
        else:
            assert 0, f'Unrecognised cpu name: {name}'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name

class WindowsPython:
    """
    Windows only. Information about installed Python with specific word size
    and version. Defaults to the currently-running Python.

    Members:

        .path:
            Path of python binary.
        .version:
            `{major}.{minor}`, e.g. `3.9` or `3.11`. Same as `version` passed
            to `__init__()` if not None, otherwise the inferred version.
        .root:
            The parent directory of `.path`; allows Python headers to be found,
            for example `{root}/include/Python.h`.
        .cpu:
            A `WindowsCpu` instance, same as `cpu` passed to `__init__()` if
            not None, otherwise the inferred cpu.

    We parse the output from `py -0p` to find all available python
    installations.
    """

    def __init__(self, cpu=None, version=None, verbose=True):
        if False:
            return 10
        "\n        Args:\n\n            cpu:\n                A WindowsCpu instance. If None, we use whatever we are running\n                on.\n            version:\n                Two-digit Python version as a string such as `3.8`. If None we\n                use current Python's version.\n            verbose:\n                If true we show diagnostics.\n        "
        if cpu is None:
            cpu = WindowsCpu(_cpu_name())
        if version is None:
            version = '.'.join(platform.python_version().split('.')[:2])
        _log(f'Looking for Python version={version!r} cpu.bits={cpu.bits!r}.')
        command = 'py -0p'
        if verbose:
            _log(f'Running: {command}')
        text = subprocess.check_output(command, shell=True, text=True)
        for line in text.split('\n'):
            m = re.match('^ *-V:([0-9.]+)(-32)? ([*])? +(.+)$', line)
            if not m:
                if verbose:
                    _log(f'No match for line={line!r}')
                continue
            version2 = m.group(1)
            bits = 32 if m.group(2) else 64
            current = m.group(3)
            if verbose:
                _log(f'version2={version2!r} bits={bits!r} from line={line!r}.')
            if bits != cpu.bits or version2 != version:
                continue
            path = m.group(4).strip()
            root = path[:path.rfind('\\')]
            if not os.path.exists(path):
                assert path.endswith('.exe'), f'path={path!r}'
                path2 = f'{path[:-4]}{version}.exe'
                _log(f'Python {path!r} does not exist; changed to: {path2!r}')
                assert os.path.exists(path2)
                path = path2
            self.path = path
            self.version = version
            self.root = root
            self.cpu = cpu
            return
        _log(f'Failed to find python matching cpu={cpu}.')
        _log(f'Output from {command!r} was:\n{text}')
        raise Exception(f'Failed to find python matching cpu={cpu}.')

    def description_ml(self, indent=''):
        if False:
            for i in range(10):
                print('nop')
        ret = textwrap.dedent(f'\n                root:    {self.root}\n                path:    {self.path}\n                version: {self.version}\n                cpu:     {self.cpu}\n                ')
        return textwrap.indent(ret, indent)

def _cpu_name():
    if False:
        i = 10
        return i + 15
    '\n    Returns `x32` or `x64` depending on Python build.\n    '
    return f'x{(32 if sys.maxsize == 2 ** 31 - 1 else 64)}'

def _log(text=''):
    if False:
        print('Hello World!')
    '\n    Logs lines with prefix.\n    '
    for line in text.split('\n'):
        print(f'{__file__}: {line}')
    sys.stdout.flush()