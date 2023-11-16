import os, stat
import os.path as op
from datetime import datetime
import shutil
from polyglot.urllib import quote
FILES_DIR = 'files'
INFO_DIR = 'info'
INFO_SUFFIX = '.trashinfo'
XDG_DATA_HOME = op.expanduser(os.environ.get('XDG_DATA_HOME', '~/.local/share'))
HOMETRASH = op.join(XDG_DATA_HOME, 'Trash')
uid = os.getuid()
TOPDIR_TRASH = '.Trash'
TOPDIR_FALLBACK = '.Trash-%s' % uid

def uniquote(raw):
    if False:
        i = 10
        return i + 15
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    return str(quote(raw))

def is_parent(parent, path):
    if False:
        print('Hello World!')
    path = op.realpath(path)
    parent = op.realpath(parent)
    return path.startswith(parent)

def format_date(date):
    if False:
        for i in range(10):
            print('nop')
    return date.strftime('%Y-%m-%dT%H:%M:%S')

def info_for(src, topdir):
    if False:
        for i in range(10):
            print('nop')
    if topdir is None or not is_parent(topdir, src):
        src = op.abspath(src)
    else:
        src = op.relpath(src, topdir)
    info = '[Trash Info]\n'
    info += 'Path=' + uniquote(src) + '\n'
    info += 'DeletionDate=' + format_date(datetime.now()) + '\n'
    return info

def check_create(dir):
    if False:
        while True:
            i = 10
    if not op.exists(dir):
        os.makedirs(dir, 448)

def trash_move(src, dst, topdir=None):
    if False:
        i = 10
        return i + 15
    filename = op.basename(src)
    filespath = op.join(dst, FILES_DIR)
    infopath = op.join(dst, INFO_DIR)
    (base_name, ext) = op.splitext(filename)
    counter = 0
    destname = filename
    while op.exists(op.join(filespath, destname)) or op.exists(op.join(infopath, destname + INFO_SUFFIX)):
        counter += 1
        destname = f'{base_name} {counter}{ext}'
    check_create(filespath)
    check_create(infopath)
    shutil.move(src, op.join(filespath, destname))
    with open(op.join(infopath, destname + INFO_SUFFIX), 'wb') as f:
        data = info_for(src, topdir)
        if not isinstance(data, bytes):
            data = data.encode('utf-8')
        f.write(data)

def find_mount_point(path):
    if False:
        for i in range(10):
            print('nop')
    path = op.realpath(path)
    while not op.ismount(path):
        path = op.split(path)[0]
    return path

def find_ext_volume_global_trash(volume_root):
    if False:
        while True:
            i = 10
    trash_dir = op.join(volume_root, TOPDIR_TRASH)
    if not op.exists(trash_dir):
        return None
    mode = os.lstat(trash_dir).st_mode
    if not op.isdir(trash_dir) or op.islink(trash_dir) or (not mode & stat.S_ISVTX):
        return None
    trash_dir = op.join(trash_dir, str(uid))
    try:
        check_create(trash_dir)
    except OSError:
        return None
    return trash_dir

def find_ext_volume_fallback_trash(volume_root):
    if False:
        return 10
    trash_dir = op.join(volume_root, TOPDIR_FALLBACK)
    check_create(trash_dir)
    return trash_dir

def find_ext_volume_trash(volume_root):
    if False:
        while True:
            i = 10
    trash_dir = find_ext_volume_global_trash(volume_root)
    if trash_dir is None:
        trash_dir = find_ext_volume_fallback_trash(volume_root)
    return trash_dir

def get_dev(path):
    if False:
        while True:
            i = 10
    return os.lstat(path).st_dev

def send2trash(path):
    if False:
        while True:
            i = 10
    if not op.exists(path):
        raise OSError('File not found: %s' % path)
    if not os.access(path, os.W_OK):
        raise OSError('Permission denied: %s' % path)
    path_dev = get_dev(path)
    trash_dev = get_dev(op.expanduser('~'))
    if path_dev == trash_dev:
        topdir = XDG_DATA_HOME
        dest_trash = HOMETRASH
    else:
        topdir = find_mount_point(path)
        trash_dev = get_dev(topdir)
        if trash_dev != path_dev:
            raise OSError("Couldn't find mount point for %s" % path)
        dest_trash = find_ext_volume_trash(topdir)
    trash_move(path, dest_trash, topdir)