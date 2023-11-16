import os.path
from PyQt6 import QtGui
from picard.const.sys import IS_WIN
if IS_WIN:
    _search_paths = []
else:
    _search_paths = [os.path.expanduser('~/.icons')]
    _search_paths += [os.path.join(path, 'icons') for path in os.environ.get('XDG_DATA_DIRS', '/usr/share').split(':')]
    _search_paths.append('/usr/share/pixmaps')
_current_theme = None
if 'XDG_CURRENT_DESKTOP' in os.environ:
    desktop = os.environ['XDG_CURRENT_DESKTOP'].lower()
    if desktop in {'gnome', 'unity'}:
        _current_theme = os.popen('gsettings get org.gnome.desktop.interface icon-theme').read().strip()[1:-1] or None
elif os.environ.get('KDE_FULL_SESSION'):
    _current_theme = os.popen('kreadconfig --file kdeglobals --group Icons --key Theme --default crystalsvg').read().strip() or None
ICON_SIZE_MENU = ('16x16',)
ICON_SIZE_TOOLBAR = ('22x22',)
ICON_SIZE_ALL = ('22x22', '16x16')

def lookup(name, size=ICON_SIZE_ALL):
    if False:
        for i in range(10):
            print('nop')
    icon = QtGui.QIcon()
    if _current_theme:
        for path in _search_paths:
            for subdir in ('actions', 'places', 'devices'):
                fullpath = os.path.join(path, _current_theme, size[0], subdir, name)
                if os.path.exists(fullpath + '.png'):
                    icon.addFile(fullpath + '.png')
                    for s in size[1:]:
                        icon.addFile(os.path.join(path, _current_theme, s, subdir, name) + '.png')
                    return icon
    for s in size:
        icon.addFile('/'.join([':', 'images', s, name]) + '.png')
    return icon