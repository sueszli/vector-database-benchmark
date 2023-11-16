import sys
import os
import re
import pathlib
base = pathlib.Path(__file__).absolute().parent.parent

def main():
    if False:
        print('Hello World!')
    with open(base / 'launcher' / 'game' / 'installer.rpy', 'w') as out:
        out.write("\n# This file imports the extensions API into the default store, and makes it\n# also contains the strings used by the extensions API, so the Ren'Py translation\n# framework can find them.\n\ninit python:\n    import installer\n\ninit python hide:\n")
        fn = base / 'launcher' / 'game' / 'installer.py'
        with open(fn) as f:
            data = f.read()
        for m in re.finditer('__?\\(".*?"\\)', data):
            out.write('    ' + m.group(0) + '\n')
if __name__ == '__main__':
    main()