"""
    speed_pefile
"""
import os
import shutil
from PyInstaller import log
from PyInstaller.building.build_main import Analysis
from PyInstaller.config import CONF
import time
from os.path import join
from tempfile import mkdtemp
logger = log.getLogger(__name__)

def speed_pefile():
    if False:
        for i in range(10):
            print('nop')
    log.logging.basicConfig(level=log.DEBUG)
    tempdir = mkdtemp('speed_pefile')
    workdir = join(tempdir, 'build')
    distdir = join(tempdir, 'dist')
    script = join(tempdir, 'speed_pefile_script.py')
    warnfile = join(workdir, 'warn.txt')
    os.makedirs(workdir)
    os.makedirs(distdir)
    with open(script, 'w') as f:
        f.write('\nfrom PySide2 import QtCore\nfrom PySide2 import QtGui\n')
    CONF['workpath'] = workdir
    CONF['distpath'] = distdir
    CONF['warnfile'] = warnfile
    CONF['hiddenimports'] = []
    CONF['spec'] = join(tempdir, 'speed_pefile_script.spec')
    CONF['specpath'] = tempdir
    CONF['specnm'] = 'speed_pefile_script'
    start = time.time()
    Analysis([script])
    duration = time.time() - start
    logger.warning('Analysis duration: %s', duration)
    shutil.rmtree(tempdir, ignore_errors=True)
if __name__ == '__main__':
    speed_pefile()