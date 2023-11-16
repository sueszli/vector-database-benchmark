import hashlib
import os
from yum.plugins import TYPE_CORE
CK_PATH = '/var/cache/salt/minion/rpmdb.cookie'
RPM_PATH = '/var/lib/rpm/Packages'
requires_api_version = '2.5'
plugin_type = TYPE_CORE

def _get_mtime():
    if False:
        return 10
    '\n    Get the modified time of the RPM Database.\n\n    Returns:\n        Unix ticks\n    '
    return os.path.exists(RPM_PATH) and int(os.path.getmtime(RPM_PATH)) or 0

def _get_checksum():
    if False:
        while True:
            i = 10
    '\n    Get the checksum of the RPM Database.\n\n    Returns:\n        hexdigest\n    '
    digest = hashlib.sha256()
    with open(RPM_PATH, 'rb') as rpm_db_fh:
        while True:
            buff = rpm_db_fh.read(4096)
            if not buff:
                break
            digest.update(buff)
    return digest.hexdigest()

def posttrans_hook(conduit):
    if False:
        return 10
    '\n    Hook after the package installation transaction.\n\n    :param conduit:\n    :return:\n    '
    if 'SALT_RUNNING' not in os.environ:
        with open(CK_PATH, 'w') as ck_fh:
            ck_fh.write('{chksum} {mtime}\n'.format(chksum=_get_checksum(), mtime=_get_mtime()))