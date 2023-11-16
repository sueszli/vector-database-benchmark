from __future__ import annotations
from stat import S_IFDIR, S_IFLNK, S_IFREG

class stat_result(dict):
    """
    stat_result: Result from stat, fstat, or lstat.

    This object provides a subset of os.stat_result attributes,
    for results returned from ObjectStoragePath.stat()

    It provides st_dev, st_ino, st_mode, st_nlink, st_uid, st_gid,
    st_size and st_mtime if they are available from the underlying
    storage. Extended attributes maybe accessed via dict access.

    See os.stat for more information.
    """
    st_dev = property(lambda self: 0)
    'device'
    st_size = property(lambda self: self._info.get('size', 0))
    'total size, in bytes'
    st_gid = property(lambda self: self._info.get('gid', 0))
    'group ID of owner'
    st_uid = property(lambda self: self._info.get('uid', 0))
    'user ID of owner'
    st_ino = property(lambda self: self._info.get('ino', 0))
    'inode'
    st_nlink = property(lambda self: self._info.get('nlink', 0))
    'number of hard links'

    @property
    def st_mtime(self):
        if False:
            return 10
        'Time of most recent content modification.'
        if 'mtime' in self:
            return self.get('mtime')
        if 'LastModified' in self:
            return self.get('LastModified').timestamp()
        return 0

    @property
    def st_mode(self):
        if False:
            while True:
                i = 10
        'Protection bits.'
        if 'mode' in self:
            return self.get('mode')
        mode = 0
        if self.get('type', '') == 'file':
            mode = S_IFREG
        if self.get('type', '') == 'directory':
            mode = S_IFDIR
        if self.get('isLink', False):
            mode = S_IFLNK
        return mode