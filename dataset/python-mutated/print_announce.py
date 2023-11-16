"""Prints release announce based on HISTORY.rst file content.
See: https://pip.pypa.io/en/stable/reference/pip_install/#hash-checking-mode.
"""
import os
import re
import subprocess
import sys
from psutil import __version__
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.realpath(os.path.join(HERE, '..', '..'))
HISTORY = os.path.join(ROOT, 'HISTORY.rst')
PRINT_HASHES_SCRIPT = os.path.join(ROOT, 'scripts', 'internal', 'print_hashes.py')
PRJ_NAME = 'psutil'
PRJ_VERSION = __version__
PRJ_URL_HOME = 'https://github.com/giampaolo/psutil'
PRJ_URL_DOC = 'http://psutil.readthedocs.io'
PRJ_URL_DOWNLOAD = 'https://pypi.org/project/psutil/#files'
PRJ_URL_WHATSNEW = 'https://github.com/giampaolo/psutil/blob/master/HISTORY.rst'
template = "Hello all,\nI'm glad to announce the release of {prj_name} {prj_version}:\n{prj_urlhome}\n\nAbout\n=====\n\npsutil (process and system utilities) is a cross-platform library for retrieving information on running processes and system utilization (CPU, memory, disks, network) in Python. It is useful mainly for system monitoring, profiling and limiting process resources and management of running processes. It implements many functionalities offered by command line tools such as: ps, top, lsof, netstat, ifconfig, who, df, kill, free, nice, ionice, iostat, iotop, uptime, pidof, tty, taskset, pmap. It currently supports Linux, Windows, macOS, Sun Solaris, FreeBSD, OpenBSD, NetBSD and AIX, both 32-bit and 64-bit architectures.  Supported Python versions are 2.7 and 3.6+. PyPy is also known to work.\n\nWhat's new\n==========\n\n{changes}\n\nLinks\n=====\n\n- Home page: {prj_urlhome}\n- Download: {prj_urldownload}\n- Documentation: {prj_urldoc}\n- What's new: {prj_urlwhatsnew}\n\nHashes\n======\n\n{hashes}\n\n--\n\nGiampaolo - https://gmpy.dev/about\n"

def get_changes():
    if False:
        i = 10
        return i + 15
    'Get the most recent changes for this release by parsing\n    HISTORY.rst file.\n    '
    with open(HISTORY) as f:
        lines = f.readlines()
    block = []
    for line in lines:
        line = lines.pop(0)
        if line.startswith('===='):
            break
    lines.pop(0)
    for line in lines:
        line = lines.pop(0)
        line = line.rstrip()
        if re.match('^- \\d+_', line):
            line = re.sub('^- (\\d+)_', '- #\\1', line)
        if line.startswith('===='):
            break
        block.append(line)
    block.pop(-1)
    while not block[-1]:
        block.pop(-1)
    return '\n'.join(block)

def main():
    if False:
        i = 10
        return i + 15
    changes = get_changes()
    hashes = subprocess.check_output([sys.executable, PRINT_HASHES_SCRIPT, 'dist/']).strip().decode()
    print(template.format(prj_name=PRJ_NAME, prj_version=PRJ_VERSION, prj_urlhome=PRJ_URL_HOME, prj_urldownload=PRJ_URL_DOWNLOAD, prj_urldoc=PRJ_URL_DOC, prj_urlwhatsnew=PRJ_URL_WHATSNEW, changes=changes, hashes=hashes))
if __name__ == '__main__':
    main()