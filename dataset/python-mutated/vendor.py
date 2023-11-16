"""Embed vcomp140.dll and msvcp140.dll."""
import os
import os.path as op
import shutil
import sys
import textwrap
TARGET_FOLDER = op.join('sklearn', '.libs')
DISTRIBUTOR_INIT = op.join('sklearn', '_distributor_init.py')
VCOMP140_SRC_PATH = 'C:\\Windows\\System32\\vcomp140.dll'
MSVCP140_SRC_PATH = 'C:\\Windows\\System32\\msvcp140.dll'

def make_distributor_init_64_bits(distributor_init, vcomp140_dll_filename, msvcp140_dll_filename):
    if False:
        i = 10
        return i + 15
    'Create a _distributor_init.py file for 64-bit architectures.\n\n    This file is imported first when importing the sklearn package\n    so as to pre-load the vendored vcomp140.dll and msvcp140.dll.\n    '
    with open(distributor_init, 'wt') as f:
        f.write(textwrap.dedent('\n            \'\'\'Helper to preload vcomp140.dll and msvcp140.dll to prevent\n            "not found" errors.\n\n            Once vcomp140.dll and msvcp140.dll are\n            preloaded, the namespace is made available to any subsequent\n            vcomp140.dll and msvcp140.dll. This is\n            created as part of the scripts that build the wheel.\n            \'\'\'\n\n\n            import os\n            import os.path as op\n            from ctypes import WinDLL\n\n\n            if os.name == "nt":\n                libs_path = op.join(op.dirname(__file__), ".libs")\n                vcomp140_dll_filename = op.join(libs_path, "{0}")\n                msvcp140_dll_filename = op.join(libs_path, "{1}")\n                WinDLL(op.abspath(vcomp140_dll_filename))\n                WinDLL(op.abspath(msvcp140_dll_filename))\n            '.format(vcomp140_dll_filename, msvcp140_dll_filename)))

def main(wheel_dirname):
    if False:
        for i in range(10):
            print('nop')
    'Embed vcomp140.dll and msvcp140.dll.'
    if not op.exists(VCOMP140_SRC_PATH):
        raise ValueError(f'Could not find {VCOMP140_SRC_PATH}.')
    if not op.exists(MSVCP140_SRC_PATH):
        raise ValueError(f'Could not find {MSVCP140_SRC_PATH}.')
    if not op.isdir(wheel_dirname):
        raise RuntimeError(f'Could not find {wheel_dirname} file.')
    vcomp140_dll_filename = op.basename(VCOMP140_SRC_PATH)
    msvcp140_dll_filename = op.basename(MSVCP140_SRC_PATH)
    target_folder = op.join(wheel_dirname, TARGET_FOLDER)
    distributor_init = op.join(wheel_dirname, DISTRIBUTOR_INIT)
    if not op.exists(target_folder):
        os.mkdir(target_folder)
    print(f'Copying {VCOMP140_SRC_PATH} to {target_folder}.')
    shutil.copy2(VCOMP140_SRC_PATH, target_folder)
    print(f'Copying {MSVCP140_SRC_PATH} to {target_folder}.')
    shutil.copy2(MSVCP140_SRC_PATH, target_folder)
    print("Generating the '_distributor_init.py' file.")
    make_distributor_init_64_bits(distributor_init, vcomp140_dll_filename, msvcp140_dll_filename)
if __name__ == '__main__':
    (_, wheel_file) = sys.argv
    main(wheel_file)