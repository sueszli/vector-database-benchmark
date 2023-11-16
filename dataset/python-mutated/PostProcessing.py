""" Postprocessing tasks for create binaries or modules.

"""
import ctypes
import os
import sys
from nuitka import Options, OutputDirectories
from nuitka.build.DataComposerInterface import getConstantBlobFilename
from nuitka.finalizations.FinalizeMarkups import getImportedNames
from nuitka.PythonVersions import getPythonABI, getTargetPythonDLLPath, python_version, python_version_str
from nuitka.Tracing import postprocessing_logger
from nuitka.utils.Execution import wrapCommandForDebuggerForExec
from nuitka.utils.FileOperations import getExternalUsePath, getFileContents, makePath, putTextFileContents, removeFileExecutablePermission
from nuitka.utils.Images import convertImageToIconFormat
from nuitka.utils.MacOSApp import createPlistInfoFile
from nuitka.utils.SharedLibraries import callInstallNameTool, cleanupHeaderForAndroid
from nuitka.utils.Utils import isAndroidBasedLinux, isMacOS, isWin32Windows
from nuitka.utils.WindowsResources import RT_GROUP_ICON, RT_ICON, RT_RCDATA, addResourceToFile, addVersionInfoResource, convertStructureToBytes, copyResourcesFromFileToFile, getDefaultWindowsExecutableManifest, getWindowsExecutableManifest

class IconDirectoryHeader(ctypes.Structure):
    _fields_ = [('reserved', ctypes.c_short), ('type', ctypes.c_short), ('count', ctypes.c_short)]

class IconDirectoryEntry(ctypes.Structure):
    _fields_ = [('width', ctypes.c_char), ('height', ctypes.c_char), ('colors', ctypes.c_char), ('reserved', ctypes.c_char), ('planes', ctypes.c_short), ('bit_count', ctypes.c_short), ('image_size', ctypes.c_int), ('image_offset', ctypes.c_int)]

class IconGroupDirectoryEntry(ctypes.Structure):
    _pack_ = 2
    _fields_ = (('width', ctypes.c_char), ('height', ctypes.c_char), ('colors', ctypes.c_char), ('reserved', ctypes.c_char), ('planes', ctypes.c_short), ('bit_count', ctypes.c_short), ('image_size', ctypes.c_int), ('id', ctypes.c_short))

def readFromFile(readable, c_struct):
    if False:
        i = 10
        return i + 15
    'Read ctypes structures from input.'
    result = c_struct()
    chunk = readable.read(ctypes.sizeof(result))
    ctypes.memmove(ctypes.byref(result), chunk, ctypes.sizeof(result))
    return result

def _addWindowsIconFromIcons(onefile):
    if False:
        i = 10
        return i + 15
    icon_group = 1
    image_id = 1
    images = []
    result_filename = OutputDirectories.getResultFullpath(onefile=onefile)
    for icon_spec in Options.getIconPaths():
        if '#' in icon_spec:
            (icon_path, icon_index) = icon_spec.rsplit('#', 1)
            icon_index = int(icon_index)
        else:
            icon_path = icon_spec
            icon_index = None
        icon_path = os.path.normcase(icon_path)
        if not icon_path.endswith('.ico'):
            postprocessing_logger.info("File '%s' is not in Windows icon format, converting to it." % icon_path)
            if icon_index is not None:
                postprocessing_logger.sysexit("Cannot specify indexes with non-ico format files in '%s'." % icon_spec)
            icon_build_path = os.path.join(OutputDirectories.getSourceDirectoryPath(onefile=onefile), 'icons')
            makePath(icon_build_path)
            converted_icon_path = os.path.join(icon_build_path, 'icon-%d.ico' % image_id)
            convertImageToIconFormat(logger=postprocessing_logger, image_filename=icon_spec, converted_icon_filename=converted_icon_path)
            icon_path = converted_icon_path
        with open(icon_path, 'rb') as icon_file:
            header = readFromFile(icon_file, IconDirectoryHeader)
            icons = [readFromFile(icon_file, IconDirectoryEntry) for _i in range(header.count)]
            if icon_index is not None:
                if icon_index > len(icons):
                    postprocessing_logger.sysexit("Error, referenced icon index %d in file '%s' with only %d icons." % (icon_index, icon_path, len(icons)))
                icons[:] = icons[icon_index:icon_index + 1]
            postprocessing_logger.info("Adding %d icon(s) from icon file '%s'." % (len(icons), icon_spec))
            for icon in icons:
                icon_file.seek(icon.image_offset, 0)
                images.append(icon_file.read(icon.image_size))
        parts = [convertStructureToBytes(header)]
        for icon in icons:
            parts.append(convertStructureToBytes(IconGroupDirectoryEntry(width=icon.width, height=icon.height, colors=icon.colors, reserved=icon.reserved, planes=icon.planes, bit_count=icon.bit_count, image_size=icon.image_size, id=image_id)))
            image_id += 1
        addResourceToFile(target_filename=result_filename, data=b''.join(parts), resource_kind=RT_GROUP_ICON, lang_id=0, res_name=icon_group, logger=postprocessing_logger)
    for (count, image) in enumerate(images, 1):
        addResourceToFile(target_filename=result_filename, data=image, resource_kind=RT_ICON, lang_id=0, res_name=count, logger=postprocessing_logger)

def executePostProcessingResources(manifest, onefile):
    if False:
        print('Hello World!')
    'Adding Windows resources to the binary.\n\n    Used for both onefile and not onefile binary, potentially two times.\n    '
    result_filename = OutputDirectories.getResultFullpath(onefile=onefile)
    if manifest is None:
        manifest = getDefaultWindowsExecutableManifest()
    if Options.shallAskForWindowsAdminRights():
        manifest.addUacAdmin()
    if Options.shallAskForWindowsUIAccessRights():
        manifest.addUacUiAccess()
    manifest.addResourceToFile(result_filename, logger=postprocessing_logger)
    if Options.getWindowsVersionInfoStrings() or Options.getProductVersion() or Options.getFileVersion():
        addVersionInfoResource(string_values=Options.getWindowsVersionInfoStrings(), product_version=Options.getProductVersion(), file_version=Options.getFileVersion(), file_date=(0, 0), is_exe=not Options.shallMakeModule(), result_filename=result_filename, logger=postprocessing_logger)
    template_exe = Options.getWindowsIconExecutablePath()
    if template_exe is not None:
        res_copied = copyResourcesFromFileToFile(template_exe, target_filename=result_filename, resource_kinds=(RT_ICON, RT_GROUP_ICON))
        if res_copied == 0:
            postprocessing_logger.warning("The specified icon template executable '%s' didn't contain anything to copy." % template_exe)
        else:
            postprocessing_logger.warning("Copied %d icon resources from '%s'." % (res_copied, template_exe))
    else:
        _addWindowsIconFromIcons(onefile=onefile)
    splash_screen_filename = Options.getWindowsSplashScreen()
    if splash_screen_filename is not None:
        splash_data = getFileContents(splash_screen_filename, mode='rb')
        addResourceToFile(target_filename=result_filename, data=splash_data, resource_kind=RT_RCDATA, lang_id=0, res_name=28, logger=postprocessing_logger)

def executePostProcessing():
    if False:
        i = 10
        return i + 15
    'Postprocessing of the resulting binary.\n\n    These are in part required steps, not usable after failure.\n    '
    result_filename = OutputDirectories.getResultFullpath(onefile=False)
    if not os.path.exists(result_filename):
        postprocessing_logger.sysexit('Error, scons failed to create the expected file %r. ' % result_filename)
    if isWin32Windows():
        if not Options.shallMakeModule():
            if python_version < 768:
                manifest = getWindowsExecutableManifest(sys.executable)
            else:
                manifest = None
            executePostProcessingResources(manifest=manifest, onefile=False)
        source_dir = OutputDirectories.getSourceDirectoryPath()
        addResourceToFile(target_filename=result_filename, data=getFileContents(getConstantBlobFilename(source_dir), mode='rb'), resource_kind=RT_RCDATA, res_name=3, lang_id=0, logger=postprocessing_logger)
    if isMacOS() and (not Options.shallMakeModule()) and (not Options.shallUseStaticLibPython()):
        python_abi_version = python_version_str + getPythonABI()
        python_dll_filename = 'libpython' + python_abi_version + '.dylib'
        python_lib_path = os.path.join(sys.prefix, 'lib')
        python_dll_path = os.path.join(python_lib_path, python_dll_filename)
        callInstallNameTool(filename=result_filename, mapping=((python_dll_filename, python_dll_path), ('@rpath/Python3.framework/Versions/%s/Python3' % python_version_str, python_dll_path)), id_path=None, rpath=python_lib_path)
    if Options.shallCreateAppBundle():
        createPlistInfoFile(logger=postprocessing_logger, onefile=False)
    if not isWin32Windows() and Options.shallMakeModule():
        removeFileExecutablePermission(result_filename)
    if isWin32Windows() and Options.shallMakeModule():
        candidate = os.path.join(os.path.dirname(result_filename), 'lib' + os.path.basename(result_filename)[:-4] + '.a')
        if os.path.exists(candidate):
            os.unlink(candidate)
    if isAndroidBasedLinux():
        cleanupHeaderForAndroid(result_filename)
    if Options.shallCreateCmdFileForExecution():
        dll_directory = getExternalUsePath(os.path.dirname(getTargetPythonDLLPath()))
        cmd_filename = OutputDirectories.getResultRunFilename(onefile=False)
        cmd_contents = '\n@echo off\nrem This script was created by Nuitka to execute \'%(exe_filename)s\' with Python DLL being found.\nset PATH=%(dll_directory)s;%%PATH%%\nset PYTHONHOME=%(python_home)s\n%(debugger_call)s"%%~dp0%(exe_filename)s" %%*\n' % {'debugger_call': ' '.join(wrapCommandForDebuggerForExec()) + ' ' if Options.shallRunInDebugger() else '', 'dll_directory': dll_directory, 'python_home': sys.prefix, 'exe_filename': os.path.basename(result_filename)}
        putTextFileContents(cmd_filename, cmd_contents)
    if Options.shallMakeModule() and Options.shallCreatePyiFile():
        pyi_filename = OutputDirectories.getResultBasePath() + '.pyi'
        putTextFileContents(filename=pyi_filename, contents='# This file was generated by Nuitka and describes the types of the\n# created shared library.\n\n# At this time it lists only the imports made and can be used by the\n# tools that bundle libraries, including Nuitka itself. For instance\n# standalone mode usage of the created library will need it.\n\n# In the future, this will also contain type information for values\n# in the module, so IDEs will use this. Therefore please include it\n# when you make software releases of the extension module that it\n# describes.\n\n%(imports)s\n\n# This is not Python source even if it looks so. Make it clear for\n# now. This was decided by PEP 484 designers.\n__name__ = ...\n\n' % {'imports': '\n'.join(('import %s' % module_name for module_name in getImportedNames()))}, encoding='utf-8')