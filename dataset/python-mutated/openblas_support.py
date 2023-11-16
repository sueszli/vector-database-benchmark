import glob
import hashlib
import os
import platform
import sysconfig
import sys
import shutil
import tarfile
import textwrap
import zipfile
from tempfile import mkstemp, gettempdir
from urllib.request import urlopen, Request
from urllib.error import HTTPError
OPENBLAS_V = '0.3.23.dev'
OPENBLAS_LONG = 'v0.3.23-293-gc2f4bdbb'
BASE_LOC = 'https://anaconda.org/scientific-python-nightly-wheels/openblas-libs'
SUPPORTED_PLATFORMS = ['linux-aarch64', 'linux-x86_64', 'musllinux-x86_64', 'linux-i686', 'linux-ppc64le', 'linux-s390x', 'win-amd64', 'win-32', 'macosx-x86_64', 'macosx-arm64']
IS_32BIT = sys.maxsize < 2 ** 32

def get_plat():
    if False:
        while True:
            i = 10
    plat = sysconfig.get_platform()
    plat_split = plat.split('-')
    arch = plat_split[-1]
    if arch == 'win32':
        plat = 'win-32'
    elif arch in ['universal2', 'intel']:
        plat = f'macosx-{platform.uname().machine}'
    elif len(plat_split) > 2:
        plat = f'{plat_split[0]}-{arch}'
    assert plat in SUPPORTED_PLATFORMS, f'invalid platform {plat}'
    return plat

def get_manylinux(arch):
    if False:
        while True:
            i = 10
    default = '2014'
    ml_ver = os.environ.get('MB_ML_VER', default)
    assert ml_ver in ('2010', '2014', '_2_24'), f'invalid MB_ML_VER {ml_ver}'
    suffix = f'manylinux{ml_ver}_{arch}.tar.gz'
    return suffix

def get_musllinux(arch):
    if False:
        return 10
    musl_ver = '1_1'
    suffix = f'musllinux_{musl_ver}_{arch}.tar.gz'
    return suffix

def get_linux(arch):
    if False:
        i = 10
        return i + 15
    try:
        from packaging.tags import sys_tags
        tags = list(sys_tags())
        plat = tags[0].platform
    except ImportError:
        plat = 'manylinux'
        v = sysconfig.get_config_var('HOST_GNU_TYPE') or ''
        if 'musl' in v:
            plat = 'musllinux'
    if 'manylinux' in plat:
        return get_manylinux(arch)
    elif 'musllinux' in plat:
        return get_musllinux(arch)

def download_openblas(target, plat, libsuffix, *, nightly=False):
    if False:
        while True:
            i = 10
    (osname, arch) = plat.split('-')
    fnsuffix = {None: '', '64_': '64_'}[libsuffix]
    filename = ''
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 ; (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
    suffix = None
    if osname == 'linux':
        suffix = get_linux(arch)
        typ = 'tar.gz'
    elif osname == 'musllinux':
        suffix = get_musllinux(arch)
        typ = 'tar.gz'
    elif plat == 'macosx-x86_64':
        suffix = 'macosx_10_9_x86_64-gf_c469a42.tar.gz'
        typ = 'tar.gz'
    elif plat == 'macosx-arm64':
        suffix = 'macosx_11_0_arm64-gf_5272328.tar.gz'
        typ = 'tar.gz'
    elif osname == 'win':
        if plat == 'win-32':
            suffix = 'win32-gcc_8_3_0.zip'
        else:
            suffix = 'win_amd64-gcc_10_3_0.zip'
        typ = 'zip'
    if not suffix:
        return None
    openblas_version = 'HEAD' if nightly else OPENBLAS_LONG
    filename = f'{BASE_LOC}/{openblas_version}/download/openblas{fnsuffix}-{openblas_version}-{suffix}'
    print(f'Attempting to download {filename}', file=sys.stderr)
    req = Request(url=filename, headers=headers)
    try:
        response = urlopen(req)
    except HTTPError:
        print(f'Could not download "{filename}"', file=sys.stderr)
        raise
    length = response.getheader('content-length')
    if response.status != 200:
        print(f'Could not download "{filename}"', file=sys.stderr)
        return None
    data = response.read()
    key = os.path.basename(filename)
    with open(target, 'wb') as fid:
        fid.write(data)
    return typ

def setup_openblas(plat=get_plat(), use_ilp64=False, nightly=False):
    if False:
        while True:
            i = 10
    '\n    Download and setup an openblas library for building. If successful,\n    the configuration script will find it automatically.\n\n    Returns\n    -------\n    msg : str\n        path to extracted files on success, otherwise indicates what went wrong\n        To determine success, do ``os.path.exists(msg)``\n    '
    if use_ilp64 and IS_32BIT:
        raise RuntimeError('Cannot use 64-bit BLAS on 32-bit arch')
    (_, tmp) = mkstemp()
    if not plat:
        raise ValueError('unknown platform')
    libsuffix = '64_' if use_ilp64 else None
    typ = download_openblas(tmp, plat, libsuffix, nightly=nightly)
    if not typ:
        return ''
    (osname, arch) = plat.split('-')
    if osname == 'win':
        if not typ == 'zip':
            return f'expecting to download zipfile on windows, not {typ}'
        return unpack_windows_zip(tmp, plat)
    else:
        if not typ == 'tar.gz':
            return 'expecting to download tar.gz, not %s' % str(typ)
        return unpack_targz(tmp)

def unpack_windows_zip(fname, plat):
    if False:
        return 10
    unzip_base = os.path.join(gettempdir(), 'openblas')
    if not os.path.exists(unzip_base):
        os.mkdir(unzip_base)
    with zipfile.ZipFile(fname, 'r') as zf:
        zf.extractall(unzip_base)
    if plat == 'win-32':
        target = os.path.join(unzip_base, '32')
    else:
        target = os.path.join(unzip_base, '64')
    lib = glob.glob(os.path.join(target, 'lib', '*.lib'))
    if len(lib) == 1:
        for f in lib:
            shutil.copy(f, os.path.join(target, 'lib', 'openblas.lib'))
            shutil.copy(f, os.path.join(target, 'lib', 'openblas64_.lib'))
    dll = glob.glob(os.path.join(target, 'bin', '*.dll'))
    for f in dll:
        shutil.copy(f, os.path.join(target, 'lib'))
    return target

def unpack_targz(fname):
    if False:
        return 10
    target = os.path.join(gettempdir(), 'openblas')
    if not os.path.exists(target):
        os.mkdir(target)
    with tarfile.open(fname, 'r') as zf:
        prefix = os.path.commonpath(zf.getnames())
        extract_tarfile_to(zf, target, prefix)
        return target

def extract_tarfile_to(tarfileobj, target_path, archive_path):
    if False:
        while True:
            i = 10
    'Extract TarFile contents under archive_path/ to target_path/'
    target_path = os.path.abspath(target_path)

    def get_members():
        if False:
            while True:
                i = 10
        for member in tarfileobj.getmembers():
            if archive_path:
                norm_path = os.path.normpath(member.name)
                if norm_path.startswith(archive_path + os.path.sep):
                    member.name = norm_path[len(archive_path) + 1:]
                else:
                    continue
            dst_path = os.path.abspath(os.path.join(target_path, member.name))
            if os.path.commonpath([target_path, dst_path]) != target_path:
                continue
            yield member
    tarfileobj.extractall(target_path, members=get_members())

def make_init(dirname):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a _distributor_init.py file for OpenBlas\n\n    Obsoleted by the use of delvewheel in wheel building, which\n    adds an equivalent snippet to numpy/__init__.py, but still useful in CI\n    '
    with open(os.path.join(dirname, '_distributor_init.py'), 'w') as fid:
        fid.write(textwrap.dedent('\n            \'\'\'\n            Helper to preload windows dlls to prevent dll not found errors.\n            Once a DLL is preloaded, its namespace is made available to any\n            subsequent DLL. This file originated in the numpy-wheels repo,\n            and is created as part of the scripts that build the wheel.\n\n            \'\'\'\n            import os\n            import glob\n            if os.name == \'nt\':\n                # load any DLL from numpy/../numpy.libs/, if present\n                try:\n                    from ctypes import WinDLL\n                except:\n                    pass\n                else:\n                    basedir = os.path.dirname(__file__)\n                    libs_dir = os.path.join(basedir, os.pardir, \'numpy.libs\')\n                    libs_dir = os.path.abspath(libs_dir)\n                    DLL_filenames = []\n                    if os.path.isdir(libs_dir):\n                        for filename in glob.glob(os.path.join(libs_dir,\n                                                               \'*openblas*dll\')):\n                            # NOTE: would it change behavior to load ALL\n                            # DLLs at this path vs. the name restriction?\n                            WinDLL(os.path.abspath(filename))\n                            DLL_filenames.append(filename)\n                    if len(DLL_filenames) > 1:\n                        import warnings\n                        warnings.warn("loaded more than 1 DLL from .libs:"\n                                      "\\n%s" % "\\n".join(DLL_filenames),\n                                      stacklevel=1)\n    '))

def test_setup(plats):
    if False:
        return 10
    '\n    Make sure all the downloadable files needed for wheel building\n    exist and can be opened\n    '

    def items():
        if False:
            return 10
        ' yields all combinations of arch, ilp64\n        '
        for plat in plats:
            yield (plat, None)
            (osname, arch) = plat.split('-')
            if arch not in ('i686', '32'):
                yield (plat, '64_')
    errs = []
    for (plat, ilp64) in items():
        (osname, _) = plat.split('-')
        if plat not in plats:
            continue
        target = None
        try:
            try:
                target = setup_openblas(plat, ilp64)
            except Exception as e:
                print(f'Could not setup {plat} with ilp64 {ilp64}, ')
                print(e)
                errs.append(e)
                continue
            if not target:
                raise RuntimeError(f'Could not setup {plat}')
            print('success with', plat, ilp64)
            files = glob.glob(os.path.join(target, 'lib', '*.a'))
            if not files:
                raise RuntimeError('No lib/*.a unpacked!')
        finally:
            if target:
                if os.path.isfile(target):
                    os.unlink(target)
                else:
                    shutil.rmtree(target)
    if errs:
        raise errs[0]

def test_version(expected_version=None):
    if False:
        print('Hello World!')
    '\n    Assert that expected OpenBLAS version is\n    actually available via NumPy. Requires threadpoolctl\n    '
    import numpy
    import threadpoolctl
    data = threadpoolctl.threadpool_info()
    if len(data) != 1:
        if platform.python_implementation() == 'PyPy':
            print(f'Not using OpenBLAS for PyPy in Azure CI, so skip this')
            return
        raise ValueError(f'expected single threadpool_info result, got {data}')
    if not expected_version:
        expected_version = OPENBLAS_V
    if data[0]['version'] != expected_version:
        raise ValueError(f'expected OpenBLAS version {expected_version}, got {data}')
    print('OK')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download and expand an OpenBLAS archive for this architecture')
    parser.add_argument('--test', nargs='*', default=None, help=f'Test different architectures. "all", or any of {SUPPORTED_PLATFORMS}')
    parser.add_argument('--check_version', nargs='?', default='', help='Check provided OpenBLAS version string against available OpenBLAS')
    parser.add_argument('--nightly', action='store_true', help='If set, use nightly OpenBLAS build.')
    parser.add_argument('--use-ilp64', action='store_true', help='If set, download the ILP64 OpenBLAS build.')
    args = parser.parse_args()
    if args.check_version != '':
        test_version(args.check_version)
    elif args.test is None:
        print(setup_openblas(nightly=args.nightly, use_ilp64=args.use_ilp64))
    elif len(args.test) == 0 or 'all' in args.test:
        test_setup(SUPPORTED_PLATFORMS)
    else:
        test_setup(args.test)