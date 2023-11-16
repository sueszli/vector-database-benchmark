from __future__ import absolute_import
import os
from trashcli.fstab.volume_of import VolumeOf

def home_trash_dir_path_from_env(environ):
    if False:
        return 10
    if 'XDG_DATA_HOME' in environ:
        return ['%(XDG_DATA_HOME)s/Trash' % environ]
    elif 'HOME' in environ:
        return ['%(HOME)s/.local/share/Trash' % environ]
    return []

def home_trash_dir_path_from_home(home_dir):
    if False:
        print('Hello World!')
    return '%s/.local/share/Trash' % home_dir

def home_trash_dir(environ, volume_of):
    if False:
        i = 10
        return i + 15
    paths = home_trash_dir_path_from_env(environ)
    for path in paths:
        yield (path, volume_of.volume_of(path))

def volume_trash_dir1(volume, uid):
    if False:
        while True:
            i = 10
    path = os.path.join(volume, '.Trash/%s' % uid)
    yield (path, volume)

def volume_trash_dir2(volume, uid):
    if False:
        print('Hello World!')
    path = os.path.join(volume, '.Trash-%s' % uid)
    yield (path, volume)