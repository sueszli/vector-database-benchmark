"""UPX plugin. """
import os
from nuitka.Options import isOnefileMode, isOnefileTempDirMode
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.AppDirs import getCacheDir
from nuitka.utils.Execution import executeToolChecked, getExecutablePath
from nuitka.utils.FileOperations import copyFile, makePath
from nuitka.utils.Hashing import Hash, getFileContentsHash
from nuitka.utils.Utils import isLinux

class NuitkaPluginUpx(NuitkaPluginBase):
    """This class represents the main logic of the UPX plugin.

    This is a plugin that removes useless stuff from DLLs and compresses the
    code at the cost of run time increases.

    """
    plugin_name = 'upx'
    plugin_desc = 'Compress created binaries with UPX automatically.'

    def __init__(self, upx_path, upx_nocache):
        if False:
            i = 10
            return i + 15
        self.upx_binary = getExecutablePath('upx', upx_path)
        self.upx_binary_hash = None
        self.upx_nocache = upx_nocache
        self.warning_given = False

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            return 10
        group.add_option('--upx-binary', action='store', dest='upx_path', default=None, help='The UPX binary to use or the directory it lives in, by default `upx` from PATH is used.')
        group.add_option('--upx-disable-cache', action='store_true', dest='upx_nocache', default=False, help='Do not cache UPX compression result, by default DLLs are cached, exe files are not.')

    @staticmethod
    def _filterUpxError(stderr):
        if False:
            i = 10
            return i + 15
        new_result = None
        if b'NotCompressibleException' in stderr or b'CantPackException' in stderr or b'AlreadyPackedException' in stderr:
            stderr = b''
            new_result = 0
        return (new_result, stderr)

    def _compressFile(self, filename, use_cache):
        if False:
            for i in range(10):
                print('nop')
        upx_options = ['-q', '--no-progress']
        if os.path.basename(filename).startswith('vcruntime140'):
            return
        if use_cache:
            if self.upx_binary_hash is None:
                self.upx_binary_hash = getFileContentsHash(self.upx_binary, as_string=False)
            upx_hash = Hash()
            upx_hash.updateFromBytes(self.upx_binary_hash)
            upx_hash.updateFromValues(*upx_options)
            upx_hash.updateFromFile(filename)
            upx_cache_dir = os.path.join(getCacheDir(), 'upx')
            makePath(upx_cache_dir)
            upx_cache_filename = os.path.join(upx_cache_dir, upx_hash.asHexDigest() + '.bin')
            if os.path.exists(upx_cache_filename):
                copyFile(upx_cache_filename, filename)
                return
        if use_cache:
            self.info("Uncached file, compressing '%s' may take a while." % os.path.basename(filename))
        else:
            self.info("Compressing '%s'." % filename)
        command = [self.upx_binary] + upx_options + [filename]
        executeToolChecked(logger=self, command=command, absence_message=None, stderr_filter=self._filterUpxError)
        if use_cache:
            copyFile(filename, upx_cache_filename)

    def _warnNoUpx(self):
        if False:
            print('Hello World!')
        if not self.warning_given:
            self.warning("No UPX binary found, please use '--upx-binary' option to specify it.")
            self.warning_given = True

    def onCopiedDLL(self, dll_filename):
        if False:
            while True:
                i = 10
        if isOnefileMode():
            pass
        elif self.upx_binary is not None:
            self._compressFile(filename=dll_filename, use_cache=not self.upx_nocache)
        else:
            self._warnNoUpx()

    def onBootstrapBinary(self, filename):
        if False:
            return 10
        if not isLinux():
            return
        if self.upx_binary is not None:
            self._compressFile(filename=filename, use_cache=False)
        else:
            self._warnNoUpx()

    def onFinalResult(self, filename):
        if False:
            print('Hello World!')
        if isLinux() and isOnefileMode():
            if not isOnefileTempDirMode():
                self.warning("UPX cannot compress '%s' as AppImage doesn't support that." % filename)
            return
        elif self.upx_binary is not None:
            self._compressFile(filename=filename, use_cache=False)
        else:
            self._warnNoUpx()