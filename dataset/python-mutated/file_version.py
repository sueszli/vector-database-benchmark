"""Get file version.
Written by Alexander Belchenko, 2006
"""
import os
import pywintypes
import win32api
__all__ = ['get_file_version', 'FileNotFound', 'VersionNotAvailable']
__docformat__ = 'restructuredtext'

class FileNotFound(Exception):
    pass

class VersionNotAvailable(Exception):
    pass

def get_file_version(filename):
    if False:
        print('Hello World!')
    'Get file version (windows properties)\n    :param  filename:   path to file\n    :return:            4-tuple with 4 version numbers\n    '
    if not os.path.isfile(filename):
        raise FileNotFound
    try:
        version_info = win32api.GetFileVersionInfo(filename, '\\')
    except pywintypes.error:
        raise VersionNotAvailable
    return divmod(version_info['FileVersionMS'], 65536) + divmod(version_info['FileVersionLS'], 65536)