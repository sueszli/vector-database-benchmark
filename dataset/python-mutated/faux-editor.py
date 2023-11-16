from __future__ import annotations
import sys
import time
import os

def main(args):
    if False:
        print('Hello World!')
    path = os.path.abspath(args[1])
    fo = open(path, 'r+')
    content = fo.readlines()
    content.append('faux editor added at %s\n' % time.time())
    fo.seek(0)
    fo.write(''.join(content))
    fo.close()
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[:]))