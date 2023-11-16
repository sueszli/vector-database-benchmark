import os
import errno
import sys

def _makedirs_31(path, exist_ok=False):
    if False:
        return 10
    try:
        os.makedirs(path)
    except OSError as exc:
        if not exist_ok or exc.errno != errno.EEXIST:
            raise
needs_makedirs = str is bytes or (3, 4) <= sys.version_info < (3, 4, 1)
makedirs = _makedirs_31 if needs_makedirs else os.makedirs