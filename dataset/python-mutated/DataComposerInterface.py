""" Interface to data composer

"""
import os
import subprocess
import sys
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Options import isExperimental
from nuitka.Tracing import data_composer_logger
from nuitka.utils.Execution import withEnvironmentVarsOverridden
from nuitka.utils.FileOperations import changeFilenameExtension, getFileSize
from nuitka.utils.Json import loadJsonFromFilename
_data_composer_size = None
_data_composer_stats = None

def getDataComposerReportValues():
    if False:
        i = 10
        return i + 15
    return OrderedDict(blob_size=_data_composer_size, stats=_data_composer_stats)

def runDataComposer(source_dir):
    if False:
        return 10
    from nuitka.plugins.Plugins import Plugins
    global _data_composer_stats
    Plugins.onDataComposerRun()
    (blob_filename, _data_composer_stats) = _runDataComposer(source_dir=source_dir)
    Plugins.onDataComposerResult(blob_filename)
    global _data_composer_size
    _data_composer_size = getFileSize(blob_filename)

def _runDataComposer(source_dir):
    if False:
        i = 10
        return i + 15
    data_composer_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'tools', 'data_composer'))
    mapping = {'NUITKA_PACKAGE_HOME': os.path.dirname(os.path.abspath(sys.modules['nuitka'].__path__[0]))}
    if isExperimental('debug-constants'):
        mapping['NUITKA_DATA_COMPOSER_VERBOSE'] = '1'
    blob_filename = getConstantBlobFilename(source_dir)
    stats_filename = changeFilenameExtension(blob_filename, '.txt')
    with withEnvironmentVarsOverridden(mapping):
        try:
            subprocess.check_call([sys.executable, data_composer_path, source_dir, blob_filename, stats_filename], shell=False)
        except subprocess.CalledProcessError:
            data_composer_logger.sysexit('Error executing data composer, please report the above exception.')
    return (blob_filename, loadJsonFromFilename(stats_filename))

def getConstantBlobFilename(source_dir):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(source_dir, '__constants.bin')

def deriveModuleConstantsBlobName(filename):
    if False:
        for i in range(10):
            print('nop')
    assert filename.endswith('.const')
    basename = filename[:-6]
    if basename == '__constants':
        return ''
    elif basename == '__bytecode':
        return '.bytecode'
    elif basename == '__files':
        return '.files'
    else:
        basename = basename[7:]
        return basename