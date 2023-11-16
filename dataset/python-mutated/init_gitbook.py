import os
import sys
import shutil

def init_data(versions):
    if False:
        while True:
            i = 10
    vs = versions.split(' ')
    for v in vs:
        src = 'node_modules'
        dst = 'docs/%s/node_modules' % v
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
if __name__ == '__main__':
    versions = sys.argv[1]
    init_data(versions)