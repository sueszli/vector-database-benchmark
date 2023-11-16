"""Purge psutil installation by removing psutil-related files and
directories found in site-packages directories. This is needed mainly
because sometimes "import psutil" imports a leftover installation
from site-packages directory instead of the main working directory.
"""
import os
import shutil
import site
PKGNAME = 'psutil'

def rmpath(path):
    if False:
        while True:
            i = 10
    if os.path.isdir(path):
        print('rmdir ' + path)
        shutil.rmtree(path)
    else:
        print('rm ' + path)
        os.remove(path)

def main():
    if False:
        for i in range(10):
            print('nop')
    locations = [site.getusersitepackages()]
    locations += site.getsitepackages()
    for root in locations:
        if os.path.isdir(root):
            for name in os.listdir(root):
                if PKGNAME in name:
                    abspath = os.path.join(root, name)
                    rmpath(abspath)
main()