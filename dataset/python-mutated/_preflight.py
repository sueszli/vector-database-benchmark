import os
import sys
from cupy_builder import Context

def preflight_check(ctx: Context) -> bool:
    if False:
        print('Hello World!')
    if sys.platform not in ('linux', 'win32'):
        print('Error: macOS is no longer supported', file=sys.stderr)
        return False
    source_root = ctx.source_root
    is_git = os.path.isdir(os.path.join(source_root, '.git'))
    for submodule in ('third_party/cccl', 'third_party/jitify', 'third_party/dlpack'):
        dirpath = os.path.join(source_root, submodule)
        if os.path.isdir(dirpath):
            if 0 < len(os.listdir(dirpath)):
                continue
        elif not is_git:
            continue
        if is_git:
            msg = f'\n===========================================================================\nThe directory {submodule} is a git submodule but is currently empty.\nPlease use the command:\n\n    $ git submodule update --init\n\nto populate the directory before building from source.\n===========================================================================\n        '
        else:
            msg = f'\n===========================================================================\nThe directory {submodule} is a git submodule but is currently empty.\nInstead of using ZIP/TAR archive downloaded from GitHub, use\n\n    $ git clone --recursive https://github.com/cupy/cupy.git\n\nto get a buildable CuPy source tree.\n===========================================================================\n        '
        print(msg, file=sys.stderr)
        return False
    return True