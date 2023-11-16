import os
import stat as st
import llnl.util.filesystem as fs
import spack.package_prefs as pp
from spack.error import SpackError

def set_permissions_by_spec(path, spec):
    if False:
        while True:
            i = 10
    if os.path.isdir(path):
        perms = pp.get_package_dir_permissions(spec)
    else:
        perms = pp.get_package_permissions(spec)
    group = pp.get_package_group(spec)
    set_permissions(path, perms, group)

def set_permissions(path, perms, group=None):
    if False:
        i = 10
        return i + 15
    perms |= os.stat(path).st_mode & (st.S_ISUID | st.S_ISGID | st.S_ISVTX)
    if perms & st.S_ISUID:
        if perms & st.S_IWOTH:
            raise InvalidPermissionsError('Attempting to set suid with world writable')
        if perms & st.S_IWGRP:
            raise InvalidPermissionsError('Attempting to set suid with group writable')
    if perms & st.S_ISGID:
        if perms & st.S_IWOTH:
            raise InvalidPermissionsError('Attempting to set sgid with world writable')
    fs.chmod_x(path, perms)
    if group:
        fs.chgrp(path, group, follow_symlinks=False)

class InvalidPermissionsError(SpackError):
    """Error class for invalid permission setters"""