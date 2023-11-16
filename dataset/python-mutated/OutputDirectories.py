""" Directories and paths to for output of Nuitka.

There are two major outputs, the build directory *.build and for
standalone mode, the *.dist folder.

A bunch of functions here are supposed to get path resolution from
this.
"""
import os
from nuitka import Options
from nuitka.utils.FileOperations import hasFilenameExtension, makePath
from nuitka.utils.Importing import getSharedLibrarySuffix
from nuitka.utils.Utils import isWin32OrPosixWindows, isWin32Windows
_main_module = None

def setMainModule(main_module):
    if False:
        while True:
            i = 10
    'Call this before using other methods of this module.'
    assert main_module.isCompiledPythonModule()
    global _main_module
    _main_module = main_module

def getSourceDirectoryPath(onefile=False):
    if False:
        for i in range(10):
            print('nop')
    'Return path inside the build directory.'
    if onefile:
        suffix = '.onefile-build'
    else:
        suffix = '.build'
    result = Options.getOutputPath(path=os.path.basename(getTreeFilenameWithSuffix(_main_module, suffix)))
    makePath(result)
    return result

def _getStandaloneDistSuffix(bundle):
    if False:
        i = 10
        return i + 15
    'Suffix to use for standalone distribution folder.'
    if bundle and Options.shallCreateAppBundle() and (not Options.isOnefileMode()):
        return '.app'
    else:
        return '.dist'

def getStandaloneDirectoryPath(bundle=True):
    if False:
        print('Hello World!')
    assert Options.isStandaloneMode()
    result = Options.getOutputPath(path=os.path.basename(getTreeFilenameWithSuffix(_main_module, _getStandaloneDistSuffix(bundle))))
    if bundle and Options.shallCreateAppBundle() and (not Options.isOnefileMode()):
        result = os.path.join(result, 'Contents', 'MacOS')
    return result

def getResultBasePath(onefile=False):
    if False:
        return 10
    if Options.isOnefileMode() and onefile:
        file_path = os.path.basename(getTreeFilenameWithSuffix(_main_module, ''))
        if Options.shallCreateAppBundle():
            file_path = os.path.join(file_path + '.app', 'Contents', 'MacOS', file_path)
        return Options.getOutputPath(path=file_path)
    elif Options.isStandaloneMode() and (not onefile):
        return os.path.join(getStandaloneDirectoryPath(), os.path.basename(getTreeFilenameWithSuffix(_main_module, '')))
    else:
        return Options.getOutputPath(path=os.path.basename(getTreeFilenameWithSuffix(_main_module, '')))

def getResultFullpath(onefile):
    if False:
        while True:
            i = 10
    'Get the final output binary result full path.'
    result = getResultBasePath(onefile=onefile)
    if Options.shallMakeModule():
        result += getSharedLibrarySuffix(preferred=True)
    else:
        output_filename = Options.getOutputFilename()
        if Options.isOnefileMode() and output_filename is not None:
            if onefile:
                result = Options.getOutputPath(output_filename)
            else:
                result = os.path.join(getStandaloneDirectoryPath(), os.path.basename(output_filename))
        elif Options.isStandaloneMode() and output_filename is not None:
            result = os.path.join(getStandaloneDirectoryPath(), os.path.basename(output_filename))
        elif output_filename is not None:
            result = output_filename
        elif not isWin32OrPosixWindows() and (not Options.shallCreateAppBundle()):
            result += '.bin'
        if isWin32OrPosixWindows() and (not hasFilenameExtension(result, '.exe')):
            result += '.exe'
    return result

def getResultRunFilename(onefile):
    if False:
        return 10
    result = getResultFullpath(onefile=onefile)
    if isWin32Windows() and Options.shallTreatUninstalledPython():
        result = getResultBasePath(onefile=onefile) + '.cmd'
    return result

def getTreeFilenameWithSuffix(module, suffix):
    if False:
        while True:
            i = 10
    return module.getOutputFilename() + suffix

def getPgoRunExecutable():
    if False:
        return 10
    return Options.getPgoExecutable() or getResultRunFilename(onefile=False)

def getPgoRunInputFilename():
    if False:
        return 10
    return getPgoRunExecutable() + '.nuitka-pgo'