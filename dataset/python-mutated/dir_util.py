"""distutils.dir_util

Utility functions for manipulating directories and directory trees."""
import os
import errno
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log
_path_created = {}

def mkpath(name, mode=511, verbose=1, dry_run=0):
    if False:
        for i in range(10):
            print('nop')
    "Create a directory and any missing ancestor directories.\n\n    If the directory already exists (or if 'name' is the empty string, which\n    means the current directory, which of course exists), then do nothing.\n    Raise DistutilsFileError if unable to create some directory along the way\n    (eg. some sub-path exists, but is a file rather than a directory).\n    If 'verbose' is true, print a one-line summary of each mkdir to stdout.\n    Return the list of directories actually created.\n    "
    global _path_created
    if not isinstance(name, str):
        raise DistutilsInternalError("mkpath: 'name' must be a string (got %r)" % (name,))
    name = os.path.normpath(name)
    created_dirs = []
    if os.path.isdir(name) or name == '':
        return created_dirs
    if _path_created.get(os.path.abspath(name)):
        return created_dirs
    (head, tail) = os.path.split(name)
    tails = [tail]
    while head and tail and (not os.path.isdir(head)):
        (head, tail) = os.path.split(head)
        tails.insert(0, tail)
    for d in tails:
        head = os.path.join(head, d)
        abs_head = os.path.abspath(head)
        if _path_created.get(abs_head):
            continue
        if verbose >= 1:
            log.info('creating %s', head)
        if not dry_run:
            try:
                os.mkdir(head, mode)
            except OSError as exc:
                if not (exc.errno == errno.EEXIST and os.path.isdir(head)):
                    raise DistutilsFileError("could not create '%s': %s" % (head, exc.args[-1]))
            created_dirs.append(head)
        _path_created[abs_head] = 1
    return created_dirs

def create_tree(base_dir, files, mode=511, verbose=1, dry_run=0):
    if False:
        for i in range(10):
            print('nop')
    "Create all the empty directories under 'base_dir' needed to put 'files'\n    there.\n\n    'base_dir' is just the name of a directory which doesn't necessarily\n    exist yet; 'files' is a list of filenames to be interpreted relative to\n    'base_dir'.  'base_dir' + the directory portion of every file in 'files'\n    will be created if it doesn't already exist.  'mode', 'verbose' and\n    'dry_run' flags are as for 'mkpath()'.\n    "
    need_dir = set()
    for file in files:
        need_dir.add(os.path.join(base_dir, os.path.dirname(file)))
    for dir in sorted(need_dir):
        mkpath(dir, mode, verbose=verbose, dry_run=dry_run)

def copy_tree(src, dst, preserve_mode=1, preserve_times=1, preserve_symlinks=0, update=0, verbose=1, dry_run=0):
    if False:
        return 10
    "Copy an entire directory tree 'src' to a new location 'dst'.\n\n    Both 'src' and 'dst' must be directory names.  If 'src' is not a\n    directory, raise DistutilsFileError.  If 'dst' does not exist, it is\n    created with 'mkpath()'.  The end result of the copy is that every\n    file in 'src' is copied to 'dst', and directories under 'src' are\n    recursively copied to 'dst'.  Return the list of files that were\n    copied or might have been copied, using their output name.  The\n    return value is unaffected by 'update' or 'dry_run': it is simply\n    the list of all files under 'src', with the names changed to be\n    under 'dst'.\n\n    'preserve_mode' and 'preserve_times' are the same as for\n    'copy_file'; note that they only apply to regular files, not to\n    directories.  If 'preserve_symlinks' is true, symlinks will be\n    copied as symlinks (on platforms that support them!); otherwise\n    (the default), the destination of the symlink will be copied.\n    'update' and 'verbose' are the same as for 'copy_file'.\n    "
    from distutils.file_util import copy_file
    if not dry_run and (not os.path.isdir(src)):
        raise DistutilsFileError("cannot copy tree '%s': not a directory" % src)
    try:
        names = os.listdir(src)
    except OSError as e:
        if dry_run:
            names = []
        else:
            raise DistutilsFileError("error listing files in '%s': %s" % (src, e.strerror))
    if not dry_run:
        mkpath(dst, verbose=verbose)
    outputs = []
    for n in names:
        src_name = os.path.join(src, n)
        dst_name = os.path.join(dst, n)
        if n.startswith('.nfs'):
            continue
        if preserve_symlinks and os.path.islink(src_name):
            link_dest = os.readlink(src_name)
            if verbose >= 1:
                log.info('linking %s -> %s', dst_name, link_dest)
            if not dry_run:
                os.symlink(link_dest, dst_name)
            outputs.append(dst_name)
        elif os.path.isdir(src_name):
            outputs.extend(copy_tree(src_name, dst_name, preserve_mode, preserve_times, preserve_symlinks, update, verbose=verbose, dry_run=dry_run))
        else:
            copy_file(src_name, dst_name, preserve_mode, preserve_times, update, verbose=verbose, dry_run=dry_run)
            outputs.append(dst_name)
    return outputs

def _build_cmdtuple(path, cmdtuples):
    if False:
        print('Hello World!')
    'Helper for remove_tree().'
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        if os.path.isdir(real_f) and (not os.path.islink(real_f)):
            _build_cmdtuple(real_f, cmdtuples)
        else:
            cmdtuples.append((os.remove, real_f))
    cmdtuples.append((os.rmdir, path))

def remove_tree(directory, verbose=1, dry_run=0):
    if False:
        for i in range(10):
            print('nop')
    "Recursively remove an entire directory tree.\n\n    Any errors are ignored (apart from being reported to stdout if 'verbose'\n    is true).\n    "
    global _path_created
    if verbose >= 1:
        log.info("removing '%s' (and everything under it)", directory)
    if dry_run:
        return
    cmdtuples = []
    _build_cmdtuple(directory, cmdtuples)
    for cmd in cmdtuples:
        try:
            cmd[0](cmd[1])
            abspath = os.path.abspath(cmd[1])
            if abspath in _path_created:
                del _path_created[abspath]
        except OSError as exc:
            log.warn('error removing %s: %s', directory, exc)

def ensure_relative(path):
    if False:
        while True:
            i = 10
    "Take the full path 'path', and make it a relative path.\n\n    This is useful to make 'path' the second argument to os.path.join().\n    "
    (drive, path) = os.path.splitdrive(path)
    if path[0:1] == os.sep:
        path = drive + path[1:]
    return path