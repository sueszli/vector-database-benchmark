import sys
import os
from vimspector import install, utils, installer

def SetUpDebugpy(wait=False, port=5678):
    if False:
        for i in range(10):
            print('nop')
    sys.path.insert(1, os.path.join(install.GetGadgetDir(utils.GetVimspectorBase()), 'debugpy', 'build', 'lib'))
    import debugpy
    exe = sys.executable
    try:
        sys.executable = installer.PathToAnyWorkingPython3()
        debugpy.listen(port)
    finally:
        sys.executable = exe
    if wait:
        debugpy.wait_for_client()