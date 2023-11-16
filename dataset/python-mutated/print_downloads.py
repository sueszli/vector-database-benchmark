"""Print PYPI statistics in MarkDown format.
Useful sites:
* https://pepy.tech/project/psutil
* https://pypistats.org/packages/psutil
* https://hugovk.github.io/top-pypi-packages/.
"""
from __future__ import print_function
import json
import os
import shlex
import subprocess
import sys
import pypinfo
from psutil._common import memoize
AUTH_FILE = os.path.expanduser('~/.pypinfo.json')
PKGNAME = 'psutil'
DAYS = 30
LIMIT = 100
GITHUB_SCRIPT_URL = 'https://github.com/giampaolo/psutil/blob/master/scripts/internal/pypistats.py'
LAST_UPDATE = None
bytes_billed = 0

@memoize
def sh(cmd):
    if False:
        for i in range(10):
            print('nop')
    assert os.path.exists(AUTH_FILE)
    env = os.environ.copy()
    env['GOOGLE_APPLICATION_CREDENTIALS'] = AUTH_FILE
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    (stdout, stderr) = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(stderr)
    assert not stderr, stderr
    return stdout.strip()

@memoize
def query(cmd):
    if False:
        for i in range(10):
            print('nop')
    global bytes_billed
    ret = json.loads(sh(cmd))
    bytes_billed += ret['query']['bytes_billed']
    return ret

def top_packages():
    if False:
        for i in range(10):
            print('nop')
    global LAST_UPDATE
    ret = query("pypinfo --all --json --days %s --limit %s '' project" % (DAYS, LIMIT))
    LAST_UPDATE = ret['last_update']
    return [(x['project'], x['download_count']) for x in ret['rows']]

def ranking():
    if False:
        print('Hello World!')
    data = top_packages()
    i = 1
    for (name, downloads) in data:
        if name == PKGNAME:
            return i
        i += 1
    raise ValueError("can't find %s" % PKGNAME)

def downloads():
    if False:
        print('Hello World!')
    data = top_packages()
    for (name, downloads) in data:
        if name == PKGNAME:
            return downloads
    raise ValueError("can't find %s" % PKGNAME)

def downloads_pyver():
    if False:
        while True:
            i = 10
    return query('pypinfo --json --days %s %s pyversion' % (DAYS, PKGNAME))

def downloads_by_country():
    if False:
        while True:
            i = 10
    return query('pypinfo --json --days %s %s country' % (DAYS, PKGNAME))

def downloads_by_system():
    if False:
        return 10
    return query('pypinfo --json --days %s %s system' % (DAYS, PKGNAME))

def downloads_by_distro():
    if False:
        print('Hello World!')
    return query('pypinfo --json --days %s %s distro' % (DAYS, PKGNAME))
templ = '| %-30s | %15s |'

def print_row(left, right):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(right, int):
        right = '{:,}'.format(right)
    print(templ % (left, right))

def print_header(left, right='Downloads'):
    if False:
        for i in range(10):
            print('nop')
    print_row(left, right)
    s = templ % ('-' * 30, '-' * 15)
    print('|:' + s[2:-2] + ':|')

def print_markdown_table(title, left, rows):
    if False:
        i = 10
        return i + 15
    pleft = left.replace('_', ' ').capitalize()
    print('### ' + title)
    print()
    print_header(pleft)
    for row in rows:
        lval = row[left]
        print_row(lval, row['download_count'])
    print()

def main():
    if False:
        while True:
            i = 10
    downs = downloads()
    print('# Download stats')
    print('')
    s = 'psutil download statistics of the last %s days (last update ' % DAYS
    s += '*%s*).\n' % LAST_UPDATE
    s += 'Generated via [pypistats.py](%s) script.\n' % GITHUB_SCRIPT_URL
    print(s)
    data = [{'what': 'Per month', 'download_count': downs}, {'what': 'Per day', 'download_count': int(downs / 30)}, {'what': 'PYPI ranking', 'download_count': ranking()}]
    print_markdown_table('Overview', 'what', data)
    print_markdown_table('Operating systems', 'system_name', downloads_by_system()['rows'])
    print_markdown_table('Distros', 'distro_name', downloads_by_distro()['rows'])
    print_markdown_table('Python versions', 'python_version', downloads_pyver()['rows'])
    print_markdown_table('Countries', 'country', downloads_by_country()['rows'])
if __name__ == '__main__':
    try:
        main()
    finally:
        print('bytes billed: %s' % bytes_billed, file=sys.stderr)