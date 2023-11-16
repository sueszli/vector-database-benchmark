import os
from calibre import patheq
from calibre.constants import iswindows
from calibre.db.legacy import LibraryDatabase
readonly = True
version = 0

def implementation(db, notify_changes):
    if False:
        while True:
            i = 10
    return (db.backend.prefs.copy(), db.backend.library_path)

def option_parser(get_parser, args):
    if False:
        print('Hello World!')
    return get_parser(_('%prog clone path/to/new/library\n\nCreate a clone of the current library. This creates a new, empty library that has all the\nsame custom columns, Virtual libraries and other settings as the current library.\n\nThe cloned library will contain no books. If you want to create a full duplicate, including\nall books, then simply use your filesystem tools to copy the library folder.\n    '))

def main(opts, args, dbctx):
    if False:
        while True:
            i = 10
    if len(args) < 1:
        raise SystemExit(_('Error: You must specify the path to the cloned library'))
    (prefs, library_path) = dbctx.run('clone')
    loc = os.path.abspath(args[0])
    if not os.path.exists(loc):
        os.makedirs(loc)
    if patheq(loc, library_path):
        raise SystemExit(_('The location for the new library is the same as the current library'))
    empty = not os.listdir(loc)
    if not empty:
        raise SystemExit(_('%s is not empty. You must choose an empty folder for the new library.') % loc)
    if iswindows and len(loc) > LibraryDatabase.WINDOWS_LIBRARY_PATH_LIMIT:
        raise SystemExit(_('Path to library too long. It must be less than %d characters.') % LibraryDatabase.WINDOWS_LIBRARY_PATH_LIMIT)
    LibraryDatabase(loc, default_prefs=prefs)
    return 0