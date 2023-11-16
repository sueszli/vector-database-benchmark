"""Platform-specific code for checking the integrity of the TensorFlow build."""
import ctypes
import os
MSVCP_DLL_NAMES = 'msvcp_dll_names'
try:
    from tensorflow.python.platform import build_info
except ImportError:
    raise ImportError('Could not import tensorflow. Do not import tensorflow from its source directory; change directory to outside the TensorFlow source tree, and relaunch your Python interpreter from there.')

def preload_check():
    if False:
        while True:
            i = 10
    'Raises an exception if the environment is not correctly configured.\n\n  Raises:\n    ImportError: If the check detects that the environment is not correctly\n      configured, and attempting to load the TensorFlow runtime will fail.\n  '
    if os.name == 'nt':
        if MSVCP_DLL_NAMES in build_info.build_info:
            missing = []
            for dll_name in build_info.build_info[MSVCP_DLL_NAMES].split(','):
                try:
                    ctypes.WinDLL(dll_name)
                except OSError:
                    missing.append(dll_name)
            if missing:
                raise ImportError('Could not find the DLL(s) %r. TensorFlow requires that these DLLs be installed in a directory that is named in your %%PATH%% environment variable. You may install these DLLs by downloading "Microsoft C++ Redistributable for Visual Studio 2015, 2017 and 2019" for your platform from this URL: https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads' % ' or '.join(missing))
    else:
        from tensorflow.python.platform import _pywrap_cpu_feature_guard
        _pywrap_cpu_feature_guard.InfoAboutUnusedCPUFeatures()