from __future__ import print_function
import os
import re
import codecs
import datetime
import pathlib
ENDINGS = ['.rpy', '.rpym', '.py', '.pyx', '.pxd']
WHITELIST = 'renpy/vc_version.py\nrenpy/angle\nrenpy/gl/__init__.py\nrenpy/common/00splines.rpy\nrenpy/common/00console.rpy\nrenpy/text/__init__.py\nmodule/maketegl.py\nmodule/generate_linebreak.py\nmodule/pysdlsound/linmixer.py\nmodule/pysdlsound/__init__.py\nmodule/build/\nmodule/include/\nmodule/gen/\nmodule/gen3/\nmodule/gen-static/\nmodule/gen3-static/\nlauncher/game/EasyDialogsResources.py\nlauncher/game/EasyDialogsWin.py\nlauncher/game/pefile.py\nlauncher/game/script_version.rpy\nlauncher/game/tl\nlauncher/game/theme'.split()
LICENSE = 'Permission is hereby granted'

def process_file(fn):
    if False:
        print('Hello World!')
    for i in ENDINGS:
        if fn.endswith(i):
            break
    else:
        return
    for i in WHITELIST:
        if fn.startswith(i):
            return
    has_copyright = False
    has_license = False
    first = True
    with open(fn, 'r') as f:
        for l in f:
            if fn.endswith('.rpy') or fn.endswith('.rpym'):
                if first:
                    if '\ufeff' not in l:
                        print('Missing BOM', fn)
                    first = False
                elif '\ufeff' in l:
                    print('Extra BOM', fn)
                first = False
            m = re.search('Copyright (\\d{4})-%d Tom Rothamel' % datetime.datetime.now().year, l)
            if m:
                has_copyright = True
            if LICENSE in l:
                has_license = True
            if has_copyright and has_license:
                return
    print('Missing copyright', fn)

def process(root):
    if False:
        print('Hello World!')
    for (dirname, _dirs, files) in os.walk(root):
        for fn in files:
            fn = os.path.join(dirname, fn)
            process_file(fn)
if __name__ == '__main__':
    os.chdir(pathlib.Path(__file__).absolute().parent.parent)
    print(os.getcwd())
    process_file('renpy.py')
    process('renpy')
    process('module')
    process('launcher/game')