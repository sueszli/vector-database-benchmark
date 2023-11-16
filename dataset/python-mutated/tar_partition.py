"""
Converting a file tree into partitioned, space-controlled TAR files.

This module attempts to address the following problems:

* Storing individual small files can be very time consuming because of
  per-file overhead.

* It is desirable to maintain UNIX metadata on a file, and that's not
  always possible without boxing the file in another format, such as
  TAR.

* Because multiple connections can allow for better throughput,
  partitioned TAR files can be parallelized for download while being
  pipelined for extraction and decompression, all to the same base
  tree.

* Ensuring that partitions are of a predictable size: the size to be
  added is bounded, as sizes must be passed up-front.  It is assumed
  that if the dataset is "hot" that supplementary write-ahead-logs
  should exist to bring the data to a consistent state.

* Representation of empty directories and symbolic links.

* Avoiding volumes with "too many" individual members to avoid
  consuming too much memory with metadata.

The *approximate* maximum size of a volume is tunable.  If any archive
members are too large, a TarMemberTooBig exception is raised: in this
case, it is necessary to raise the partition size.  The volume size
does *not* include Tar metadata overhead, and this is why one cannot
rely on an exact maximum (without More Programming).

Why not GNU Tar with its multi-volume functionality: it's relatively
difficult to limit the size of an archive member (a problem for fast
growing files that are also being WAL-logged), and GNU Tar uses
interactive prompts to ask for the right tar file to continue the next
extraction.  This coupling between tarfiles makes the extraction
process considerably more complicated.

"""
import collections
import errno
import os
import tarfile
import sys
from wal_e import files
from wal_e import log_help
from wal_e import copyfileobj
from wal_e import pipebuf
from wal_e import pipeline
from wal_e.exception import UserException
logger = log_help.WalELogger(__name__)
PG_CONF = ('postgresql.conf', 'pg_hba.conf', 'recovery.conf', 'recovery.done', 'pg_ident.conf', 'promote')

class StreamPadFileObj(object):
    """
    Layer on a file to provide a precise stream byte length

    This file-like-object accepts an underlying file-like-object and a
    target size.  Once the target size is reached, no more bytes will
    be returned.  Furthermore, if the underlying stream runs out of
    bytes, '\x00' will be returned until the target size is reached.

    """
    __slots__ = ('underlying_fp', 'target_size', 'pos')

    def __init__(self, underlying_fp, target_size):
        if False:
            i = 10
            return i + 15
        self.underlying_fp = underlying_fp
        self.target_size = target_size
        self.pos = 0

    def read(self, size):
        if False:
            return 10
        max_readable = min(self.target_size - self.pos, size)
        ret = self.underlying_fp.read(max_readable)
        lenret = len(ret)
        self.pos += lenret
        return ret + b'\x00' * (max_readable - lenret)

    def close(self):
        if False:
            return 10
        return self.underlying_fp.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.close()
        return False

class TarMemberTooBigError(UserException):

    def __init__(self, member_name, limited_to, requested, *args, **kwargs):
        if False:
            return 10
        self.member_name = member_name
        self.max_size = limited_to
        self.requested = requested
        msg = 'Attempted to archive a file that is too large.'
        hint = 'There is a file in the postgres database directory that is larger than %d bytes. If no such file exists, please report this as a bug. In particular, check %s, which appears to be %d bytes.' % (limited_to, member_name, requested)
        UserException.__init__(self, *args, msg=msg, hint=hint, **kwargs)

class TarBadRootError(Exception):

    def __init__(self, root, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.root = root
        Exception.__init__(self, *args, **kwargs)

class TarBadPathError(Exception):
    """
    Raised when a root directory does not contain all file paths.

    """

    def __init__(self, root, offensive_path, *args, **kwargs):
        if False:
            return 10
        self.root = root
        self.offensive_path = offensive_path
        Exception.__init__(self, *args, **kwargs)
ExtendedTarInfo = collections.namedtuple('ExtendedTarInfo', 'submitted_path tarinfo')
PARTITION_MAX_SZ = 1610612736
PARTITION_MAX_MEMBERS = int(PARTITION_MAX_SZ / 262144)

def _fsync_files(filenames):
    if False:
        i = 10
        return i + 15
    'Call fsync() a list of file names\n\n    The filenames should be absolute paths already.\n\n    '
    touched_directories = set()
    mode = os.O_RDONLY
    if hasattr(os, 'O_BINARY'):
        mode |= os.O_BINARY
    for filename in filenames:
        fd = os.open(filename, mode)
        os.fsync(fd)
        os.close(fd)
        touched_directories.add(os.path.dirname(filename))
    if hasattr(os, 'O_DIRECTORY'):
        for dirname in touched_directories:
            fd = os.open(dirname, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)

def cat_extract(tar, member, targetpath):
    if False:
        while True:
            i = 10
    'Extract a regular file member using cat for async-like I/O\n\n    Mostly adapted from tarfile.py.\n\n    '
    assert member.isreg()
    targetpath = targetpath.rstrip('/')
    targetpath = targetpath.replace('/', os.sep)
    upperdirs = os.path.dirname(targetpath)
    if upperdirs and (not os.path.exists(upperdirs)):
        try:
            os.makedirs(upperdirs)
        except EnvironmentError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
    with files.DeleteOnError(targetpath) as dest:
        with pipeline.get_cat_pipeline(pipeline.PIPE, dest.f) as pl:
            fp = tar.extractfile(member)
            copyfileobj.copyfileobj(fp, pl.stdin)
    if sys.version_info < (3, 5):
        tar.chown(member, targetpath)
    else:
        tar.chown(member, targetpath, False)
    tar.chmod(member, targetpath)
    tar.utime(member, targetpath)

class TarPartition(list):

    def __init__(self, name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.name = name
        list.__init__(self, *args, **kwargs)

    @staticmethod
    def _padded_tar_add(tar, et_info):
        if False:
            print('Hello World!')
        try:
            with open(et_info.submitted_path, 'rb') as raw_file:
                with StreamPadFileObj(raw_file, et_info.tarinfo.size) as f:
                    tar.addfile(et_info.tarinfo, f)
        except EnvironmentError as e:
            if e.errno == errno.ENOENT and e.filename == et_info.submitted_path:
                logger.debug(msg='tar member additions skipping an unlinked file', detail='Skipping {0}.'.format(et_info.submitted_path))
            else:
                raise

    @staticmethod
    def tarfile_extract(fileobj, dest_path):
        if False:
            i = 10
            return i + 15
        'Extract a tarfile described by a file object to a specified path.\n\n        Args:\n            fileobj (file): File object wrapping the target tarfile.\n            dest_path (str): Path to extract the contents of the tarfile to.\n        '
        tar = tarfile.open(mode='r|', fileobj=fileobj, bufsize=pipebuf.PIPE_BUF_BYTES)
        dest_path = os.path.realpath(dest_path)
        extracted_files = []
        for member in tar:
            assert not member.name.startswith('/')
            relpath = os.path.join(dest_path, member.name)
            if member.issym():
                target_path = os.path.join(dest_path, member.name)
                try:
                    os.symlink(member.linkname, target_path)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        os.remove(target_path)
                        os.symlink(member.linkname, target_path)
                    else:
                        raise
                continue
            if member.isreg() and member.size >= pipebuf.PIPE_BUF_BYTES:
                cat_extract(tar, member, relpath)
            else:
                tar.extract(member, path=dest_path)
            filename = os.path.realpath(relpath)
            extracted_files.append(filename)
            if len(extracted_files) > 1000:
                _fsync_files(extracted_files)
                del extracted_files[:]
        tar.close()
        _fsync_files(extracted_files)

    def tarfile_write(self, fileobj):
        if False:
            while True:
                i = 10
        tar = None
        try:
            tar = tarfile.open(fileobj=fileobj, mode='w|', bufsize=pipebuf.PIPE_BUF_BYTES)
            for et_info in self:
                if et_info.tarinfo.isfile():
                    self._padded_tar_add(tar, et_info)
                else:
                    tar.addfile(et_info.tarinfo)
        finally:
            if tar is not None:
                tar.close()

    @property
    def total_member_size(self):
        if False:
            while True:
                i = 10
        '\n        Compute the sum of the size of expanded TAR member\n\n        Expressed in bytes.\n\n        '
        return sum((et_info.tarinfo.size for et_info in self))

    def format_manifest(self):
        if False:
            i = 10
            return i + 15
        parts = []
        for tpart in self:
            for et_info in tpart:
                tarinfo = et_info.tarinfo
                parts.append('\t'.join([tarinfo.name, tarinfo.size]))
        return '\n'.join(parts)

def _segmentation_guts(root, file_paths, max_partition_size):
    if False:
        i = 10
        return i + 15
    'Segment a series of file paths into TarPartition values\n\n    These TarPartitions are disjoint and roughly below the prescribed\n    size.\n    '
    if not root.endswith(os.path.sep):
        root += os.path.sep
    if not os.path.isdir(root):
        raise TarBadRootError(root=root)
    bogus_tar = None
    try:
        bogus_tar = tarfile.TarFile(os.devnull, 'w', dereference=False)
        partition_number = 0
        partition_bytes = 0
        partition_members = 0
        partition = TarPartition(partition_number)
        for file_path in file_paths:
            if not file_path.startswith(root):
                raise TarBadPathError(root=root, offensive_path=file_path)
            try:
                et_info = ExtendedTarInfo(tarinfo=bogus_tar.gettarinfo(file_path, arcname=file_path[len(root):]), submitted_path=file_path)
            except EnvironmentError as e:
                if e.errno == errno.ENOENT and e.filename == file_path:
                    logger.debug(msg='tar member additions skipping an unlinked file', detail='Skipping {0}.'.format(et_info.submitted_path))
                    continue
                else:
                    raise
            if et_info.tarinfo.size > max_partition_size:
                raise TarMemberTooBigError(et_info.tarinfo.name, max_partition_size, et_info.tarinfo.size)
            if partition_bytes + et_info.tarinfo.size >= max_partition_size or partition_members >= PARTITION_MAX_MEMBERS:
                yield partition
                partition_number += 1
                partition_bytes = et_info.tarinfo.size
                partition_members = 1
                partition = TarPartition(partition_number, [et_info])
            else:
                partition_bytes += et_info.tarinfo.size
                partition_members += 1
                partition.append(et_info)
                assert partition_bytes < max_partition_size
    finally:
        if bogus_tar is not None:
            bogus_tar.close()
    if partition:
        yield partition

def do_not_descend(root, name, dirnames, matches):
    if False:
        return 10
    if name in dirnames:
        dirnames.remove(name)
        matches.append(os.path.join(root, name))

def partition(pg_cluster_dir):
    if False:
        i = 10
        return i + 15

    def raise_walk_error(e):
        if False:
            i = 10
            return i + 15
        raise e
    if not pg_cluster_dir.endswith(os.path.sep):
        pg_cluster_dir += os.path.sep
    matches = []
    spec = {'base_prefix': pg_cluster_dir, 'tablespaces': []}
    walker = os.walk(pg_cluster_dir, onerror=raise_walk_error)
    for (root, dirnames, filenames) in walker:
        is_cluster_toplevel = os.path.abspath(root) == os.path.abspath(pg_cluster_dir)
        matches.append(root)
        if is_cluster_toplevel:
            for name in ['pg_xlog', 'pg_log', 'pg_replslot', 'pg_wal']:
                do_not_descend(root, name, dirnames, matches)
        for name in ['pgsql_tmp', 'pg_stat_tmp', '.wal-e']:
            do_not_descend(root, name, dirnames, matches)
        if 'lost+found' in dirnames:
            dirnames.remove('lost+found')
        for filename in filenames:
            if is_cluster_toplevel and filename in ('postmaster.pid', 'postmaster.opts'):
                pass
            elif is_cluster_toplevel and filename in PG_CONF:
                pass
            else:
                matches.append(os.path.join(root, filename))
        if root == os.path.join(pg_cluster_dir, 'pg_tblspc'):
            for tablespace in dirnames:
                ts_path = os.path.join(root, tablespace)
                ts_name = os.path.basename(ts_path)
                if os.path.islink(ts_path) and os.path.isdir(ts_path):
                    ts_loc = os.readlink(ts_path)
                    ts_walker = os.walk(ts_path)
                    if not ts_loc.endswith(os.path.sep):
                        ts_loc += os.path.sep
                    if ts_name not in spec['tablespaces']:
                        spec['tablespaces'].append(ts_name)
                        link_start = len(spec['base_prefix'])
                        spec[ts_name] = {'loc': ts_loc, 'link': ts_path[link_start:]}
                    for (ts_root, ts_dirnames, ts_filenames) in ts_walker:
                        if 'pgsql_tmp' in ts_dirnames:
                            ts_dirnames.remove('pgsql_tmp')
                            matches.append(os.path.join(ts_root, 'pgsql_tmp'))
                        for ts_filename in ts_filenames:
                            matches.append(os.path.join(ts_root, ts_filename))
                        if not ts_filenames and ts_root not in matches:
                            matches.append(ts_root)
                    if ts_path in matches:
                        matches.remove(ts_path)
    local_abspaths = [os.path.abspath(match) for match in matches]
    local_prefix = os.path.commonprefix(local_abspaths)
    if not local_prefix.endswith(os.path.sep):
        local_prefix += os.path.sep
    parts = _segmentation_guts(local_prefix, matches, PARTITION_MAX_SZ)
    return (spec, parts)