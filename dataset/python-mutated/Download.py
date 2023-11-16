""" Download utilities and extract locally when allowed.

Mostly used on Windows, for dependency walker and ccache binaries.
"""
import os
from nuitka import Tracing
from nuitka.__past__ import urlretrieve
from nuitka.Progress import withNuitkaDownloadProgressBar
from .AppDirs import getCacheDir
from .FileOperations import addFileExecutablePermission, deleteFile, makePath, queryUser

def getDownload(name, url, download_path):
    if False:
        while True:
            i = 10
    with withNuitkaDownloadProgressBar(desc='Download %s' % name) as reporthook:
        try:
            try:
                urlretrieve(url, download_path, reporthook=reporthook)
            except Exception:
                urlretrieve(url.replace('https://', 'http://'), download_path, reporthook=reporthook)
        except KeyboardInterrupt:
            deleteFile(download_path, must_exist=False)
            raise

def getDownloadCacheDir():
    if False:
        return 10
    return os.path.join(getCacheDir(), 'downloads')

def getCachedDownload(name, url, binary, flatten, is_arch_specific, specificity, message, reject, assume_yes_for_downloads):
    if False:
        for i in range(10):
            print('nop')
    nuitka_download_dir = getDownloadCacheDir()
    nuitka_download_dir = os.path.join(nuitka_download_dir, os.path.basename(binary).replace('.exe', ''))
    if is_arch_specific:
        nuitka_download_dir = os.path.join(nuitka_download_dir, is_arch_specific)
    if specificity:
        nuitka_download_dir = os.path.join(nuitka_download_dir, specificity)
    download_path = os.path.join(nuitka_download_dir, os.path.basename(url))
    exe_path = os.path.join(nuitka_download_dir, binary)
    makePath(nuitka_download_dir)
    if not os.path.isfile(download_path) and (not os.path.isfile(exe_path)):
        if assume_yes_for_downloads:
            reply = 'yes'
        else:
            reply = queryUser(question="%s\n\nIs it OK to download and put it in '%s'.\n\nFully automatic, cached. Proceed and download" % (message, nuitka_download_dir), choices=('yes', 'no'), default='yes', default_non_interactive='no')
        if reply != 'yes':
            if reject is not None:
                Tracing.general.sysexit(reject)
        else:
            Tracing.general.info("Downloading '%s'." % url)
            try:
                getDownload(name=name, url=url, download_path=download_path)
            except Exception as e:
                Tracing.general.sysexit("Failed to download '%s' due to '%s'. Contents should manually be copied to '%s'." % (url, e, download_path))
    if not os.path.isfile(exe_path) and os.path.isfile(download_path):
        Tracing.general.info("Extracting to '%s'" % exe_path)
        import zipfile
        try:
            zip_file = zipfile.ZipFile(download_path)
            for zip_info in zip_file.infolist():
                if zip_info.filename[-1] == '/':
                    continue
                if flatten:
                    zip_info.filename = os.path.basename(zip_info.filename)
                zip_file.extract(zip_info, nuitka_download_dir)
        except Exception:
            Tracing.general.info('Problem with the downloaded zip file, deleting it.')
            deleteFile(binary, must_exist=False)
            deleteFile(download_path, must_exist=True)
            Tracing.general.sysexit("Error, need '%s' as extracted from '%s'." % (binary, url))
    if os.path.isfile(exe_path):
        addFileExecutablePermission(exe_path)
    else:
        if reject:
            Tracing.general.sysexit(reject)
        exe_path = None
    return exe_path

def getCachedDownloadedMinGW64(target_arch, assume_yes_for_downloads):
    if False:
        while True:
            i = 10
    if target_arch == 'x86_64':
        url = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/13.2.0-16.0.6-11.0.1-msvcrt-r1/winlibs-x86_64-posix-seh-gcc-13.2.0-llvm-16.0.6-mingw-w64msvcrt-11.0.1-r1.zip'
        binary = 'mingw64\\bin\\gcc.exe'
    elif target_arch == 'x86':
        url = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/13.2.0-16.0.6-11.0.1-msvcrt-r1/winlibs-i686-posix-dwarf-gcc-13.2.0-llvm-16.0.6-mingw-w64msvcrt-11.0.1-r1.zip'
        binary = 'mingw32\\bin\\gcc.exe'
    elif target_arch == 'arm64':
        url = None
    else:
        assert False, target_arch
    if url is None:
        return None
    gcc_binary = getCachedDownload(name='mingw64', url=url, is_arch_specific=target_arch, specificity=url.rsplit('/', 2)[1], binary=binary, flatten=False, message='Nuitka will use gcc from MinGW64 of winlibs to compile on Windows.', reject='Only this specific gcc is supported with Nuitka.', assume_yes_for_downloads=assume_yes_for_downloads)
    return gcc_binary