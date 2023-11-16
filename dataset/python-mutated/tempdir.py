import os
import re
from golem.core.common import is_osx

def fix_osx_tmpdir() -> bool:
    if False:
        return 10
    if not is_osx():
        return False
    tmpdir = os.environ.get('TMPDIR') or ''
    if not re.match('^/(private|tmp|Users|Volumes).*', tmpdir):
        os.environ['TMPDIR'] = '/tmp'
        print(f"\x1b[0;31mTMPDIR updated to something that Docker can mount: {os.environ.get('TMPDIR')}\x1b[0m")
        return True
    return False