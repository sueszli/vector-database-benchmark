from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import os
import zipfile
import json
import renpy
import threading
from renpy.loadsave import clear_slot, safe_rename
import shutil
disk_lock = threading.RLock()
import time
tmp = '.' + str(int(time.time())) + '.tmp'

class FileLocation(object):
    """
    A location that saves files to a directory on disk.
    """

    def __init__(self, directory):
        if False:
            for i in range(10):
                print('nop')
        self.directory = directory
        try:
            os.makedirs(self.directory)
        except Exception:
            pass
        renpy.util.expose_directory(self.directory)
        try:
            fn = os.path.join(self.directory, 'text.txt')
            with open(fn, 'w') as f:
                f.write('Test.')
            os.unlink(fn)
            self.active = True
        except Exception:
            self.active = False
        self.mtimes = {}
        self.persistent = os.path.join(self.directory, 'persistent')
        self.persistent_mtime = 0
        self.persistent_data = None

    def filename(self, slotname):
        if False:
            while True:
                i = 10
        '\n        Given a slot name, returns a filename.\n        '
        return os.path.join(self.directory, renpy.exports.fsencode(slotname + renpy.savegame_suffix))

    def sync(self):
        if False:
            return 10
        '\n        Called to indicate that the HOME filesystem was changed.\n        '
        if renpy.emscripten:
            import emscripten
            emscripten.syncfs()

    def scan(self):
        if False:
            return 10
        '\n        Scan for files that are added or removed.\n        '
        if not self.active:
            return
        with disk_lock:
            old_mtimes = self.mtimes
            new_mtimes = {}
            suffix = renpy.savegame_suffix
            suffix_len = len(suffix)
            for fn in os.listdir(self.directory):
                if not fn.endswith(suffix):
                    continue
                slotname = fn[:-suffix_len]
                try:
                    new_mtimes[slotname] = os.path.getmtime(os.path.join(self.directory, fn))
                except Exception:
                    pass
            self.mtimes = new_mtimes
            for (slotname, mtime) in new_mtimes.items():
                if old_mtimes.get(slotname, None) != mtime:
                    clear_slot(slotname)
            for slotname in old_mtimes:
                if slotname not in new_mtimes:
                    clear_slot(slotname)
            for pfn in [self.persistent + '.new', self.persistent]:
                if os.path.exists(pfn):
                    mtime = os.path.getmtime(pfn)
                    if mtime != self.persistent_mtime:
                        data = renpy.persistent.load(pfn)
                        if data is not None:
                            self.persistent_mtime = mtime
                            self.persistent_data = data
                            break

    def save(self, slotname, record):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the save record in slotname.\n        '
        filename = self.filename(slotname)
        with disk_lock:
            record.write_file(filename)
        renpy.util.expose_file(filename)
        self.sync()
        self.scan()

    def list(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of all slots with savefiles in them, in arbitrary\n        order.\n        '
        return list(self.mtimes)

    def list_files(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of all the actual save files.\n        '
        rv = []
        for slotname in self.list():
            rv.append(self.filename(slotname))
        return rv

    def mtime(self, slotname):
        if False:
            i = 10
            return i + 15
        '\n        For a slot, returns the time the object was saved in that\n        slot.\n\n        Returns None if the slot is empty.\n        '
        return self.mtimes.get(slotname, None)

    def path(self, filename):
        if False:
            return 10
        '\n        Returns the mtime and path of the given filename, or (0, None) if\n        the file does not exist.\n        '
        with disk_lock:
            fn = os.path.join(self.directory, filename)
            try:
                return (os.path.getmtime(fn), fn)
            except Exception:
                return (0, None)

    def json(self, slotname):
        if False:
            i = 10
            return i + 15
        '\n        Returns the JSON data for slotname.\n\n        Returns None if the slot is empty.\n        '
        with disk_lock:
            try:
                filename = self.filename(slotname)
                with zipfile.ZipFile(filename, 'r') as zf:
                    try:
                        data = zf.read('json')
                        data = json.loads(data)
                        return data
                    except Exception:
                        pass
                    try:
                        extra_info = zf.read('extra_info').decode('utf-8')
                        return {'_save_name': extra_info}
                    except Exception:
                        pass
                    return {}
            except Exception:
                return None

    def screenshot(self, slotname):
        if False:
            i = 10
            return i + 15
        '\n        Returns a displayable that show the screenshot for this slot.\n\n        Returns None if the slot is empty.\n        '
        with disk_lock:
            mtime = self.mtime(slotname)
            if mtime is None:
                return None
            try:
                filename = self.filename(slotname)
                with zipfile.ZipFile(filename, 'r') as zf:
                    try:
                        png = False
                        zf.getinfo('screenshot.tga')
                    except Exception:
                        png = True
                        zf.getinfo('screenshot.png')
            except Exception:
                return None
            if png:
                screenshot = renpy.display.im.ZipFileImage(filename, 'screenshot.png', mtime)
            else:
                screenshot = renpy.display.im.ZipFileImage(filename, 'screenshot.tga', mtime)
            return screenshot

    def load(self, slotname):
        if False:
            return 10
        '\n        Returns the log and signature components of the file found in `slotname`\n        '
        with disk_lock:
            filename = self.filename(slotname)
            with zipfile.ZipFile(filename, 'r') as zf:
                log = zf.read('log')
                try:
                    token = zf.read('signatures').decode('utf-8')
                except:
                    token = ''
            return (log, token)

    def unlink(self, slotname):
        if False:
            i = 10
            return i + 15
        '\n        Deletes the file in slotname.\n        '
        with disk_lock:
            filename = self.filename(slotname)
            if os.path.exists(filename):
                os.unlink(filename)
            self.sync()
            self.scan()

    def rename(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        '\n        If old exists, renames it to new.\n        '
        with disk_lock:
            old = self.filename(old)
            new = self.filename(new)
            if not os.path.exists(old):
                return
            old_tmp = old + tmp
            safe_rename(old, old_tmp)
            safe_rename(old_tmp, new)
            renpy.util.expose_file(new)
            self.sync()
            self.scan()

    def copy(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        '\n        Copies `old` to `new`, if `old` exists.\n        '
        with disk_lock:
            old = self.filename(old)
            new = self.filename(new)
            if not os.path.exists(old):
                return
            shutil.copyfile(old, new)
            renpy.util.expose_file(new)
            self.sync()
            self.scan()

    def load_persistent(self):
        if False:
            print('Hello World!')
        '\n        Returns a list of (mtime, persistent) tuples loaded from the\n        persistent file. This should return quickly, with the actual\n        load occuring in the scan thread.\n        '
        if self.persistent_data:
            return [(self.persistent_mtime, self.persistent_data)]
        else:
            return []

    def save_persistent(self, data):
        if False:
            print('Hello World!')
        '\n        Saves `data` as the persistent data. Data is a binary string giving\n        the persistent data in python format.\n        '
        with disk_lock:
            if not self.active:
                return
            fn = self.persistent
            fn_tmp = fn + tmp
            fn_new = fn + '.new'
            with open(fn_tmp, 'wb') as f:
                f.write(data)
            safe_rename(fn_tmp, fn_new)
            safe_rename(fn_new, fn)
            self.persistent_mtime = os.path.getmtime(fn)
            renpy.util.expose_file(fn)
            self.sync()

    def unlink_persistent(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.active:
            return
        try:
            os.unlink(self.persistent)
            self.sync()
        except Exception:
            pass

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, FileLocation):
            return False
        return self.directory == other.directory

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

class MultiLocation(object):
    """
    A location that saves in multiple places. When loading or otherwise
    accessing a file, it loads the newest file found for the given slotname.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.locations = []

    def active_locations(self):
        if False:
            i = 10
            return i + 15
        return [i for i in self.locations if i.active]

    def newest(self, slotname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the location containing the slotname with the newest\n        mtime. Returns None if the slot is empty.\n        '
        if not renpy.config.save:
            return None
        mtime = -1
        location = None
        for l in self.locations:
            if not l.active:
                continue
            slot_mtime = l.mtime(slotname)
            if slot_mtime is not None:
                if slot_mtime > mtime:
                    mtime = slot_mtime
                    location = l
        return location

    def add(self, location):
        if False:
            while True:
                i = 10
        '\n        Adds a new location.\n        '
        if location in self.locations:
            return
        self.locations.append(location)

    def save(self, slotname, record):
        if False:
            while True:
                i = 10
        if not renpy.config.save:
            return
        saved = False
        for l in self.active_locations():
            l.save(slotname, record)
            saved = True
        if not saved:
            raise Exception('Not saved - no valid save locations.')

    def list(self):
        if False:
            return 10
        if not renpy.config.save:
            return []
        rv = set()
        for l in self.active_locations():
            rv.update(l.list())
        return list(rv)

    def list_files(self):
        if False:
            print('Hello World!')
        if not renpy.config.save:
            return []
        rv = []
        for l in self.active_locations():
            rv.extend(l.list_files())
        return rv

    def path(self, filename):
        if False:
            i = 10
            return i + 15
        results = []
        for i in self.active_locations():
            results.append(i.path(filename))
        if not results:
            return (0, None)
        results.sort()
        return results[-1]

    def mtime(self, slotname):
        if False:
            i = 10
            return i + 15
        l = self.newest(slotname)
        if l is None:
            return None
        return l.mtime(slotname)

    def json(self, slotname):
        if False:
            for i in range(10):
                print('nop')
        l = self.newest(slotname)
        if l is None:
            return None
        return l.json(slotname)

    def screenshot(self, slotname):
        if False:
            while True:
                i = 10
        l = self.newest(slotname)
        if l is None:
            return None
        return l.screenshot(slotname)

    def load(self, slotname):
        if False:
            i = 10
            return i + 15
        l = self.newest(slotname)
        return l.load(slotname)

    def unlink(self, slotname):
        if False:
            print('Hello World!')
        if not renpy.config.save:
            return
        for l in self.active_locations():
            l.unlink(slotname)

    def rename(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        if not renpy.config.save:
            return
        for l in self.active_locations():
            l.rename(old, new)

    def copy(self, old, new):
        if False:
            while True:
                i = 10
        if not renpy.config.save:
            return
        for l in self.active_locations():
            l.copy(old, new)

    def load_persistent(self):
        if False:
            i = 10
            return i + 15
        rv = []
        for l in self.active_locations():
            rv.extend(l.load_persistent())
        return rv

    def save_persistent(self, data):
        if False:
            return 10
        for l in self.active_locations():
            l.save_persistent(data)

    def unlink_persistent(self):
        if False:
            print('Hello World!')
        for l in self.active_locations():
            l.unlink_persistent()

    def scan(self):
        if False:
            while True:
                i = 10
        for l in self.locations:
            l.scan()

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, MultiLocation):
            return False
        return self.locations == other.locations

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other
scan_thread = None
quit_scan_thread = False
scan_thread_condition = threading.Condition()

def run_scan_thread():
    if False:
        return 10
    global quit_scan_thread
    quit_scan_thread = False
    while not quit_scan_thread:
        try:
            renpy.loadsave.location.scan()
        except Exception:
            pass
        with scan_thread_condition:
            scan_thread_condition.wait(5.0)

def quit():
    if False:
        for i in range(10):
            print('nop')
    global quit_scan_thread
    with scan_thread_condition:
        quit_scan_thread = True
        scan_thread_condition.notify_all()
    if scan_thread is not None:
        scan_thread.join()

def init():
    if False:
        while True:
            i = 10
    global scan_thread
    global quit_scan_thread
    quit()
    quit_scan_thread = False
    location = MultiLocation()
    location.add(FileLocation(renpy.config.savedir))
    if not renpy.mobile and (not renpy.macapp):
        path = os.path.join(renpy.config.gamedir, 'saves')
        location.add(FileLocation(path))
    for i in renpy.config.extra_savedirs:
        location.add(FileLocation(i))
    location.scan()
    renpy.loadsave.location = location
    if not renpy.emscripten:
        scan_thread = threading.Thread(target=run_scan_thread)
        scan_thread.start()

def zip_saves():
    if False:
        for i in range(10):
            print('nop')
    '\n    This is called directly from Javascript, to zip up the savegames\n    to /savegames.zip.\n    '
    import zipfile
    import pathlib
    p = pathlib.Path(renpy.config.savedir)
    with zipfile.ZipFile('savegames.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in p.rglob('*'):
            zf.write(fn, fn.relative_to(p))
    return True

def unzip_saves():
    if False:
        while True:
            i = 10
    import zipfile
    import pathlib
    p = pathlib.Path(renpy.config.savedir)
    with zipfile.ZipFile('savegames.zip', 'r') as zf:
        for i in zf.infolist():
            if '/' not in i.filename:
                filename = i.filename
            else:
                (prefix, _, filename) = i.filename.partition('/')
                if not renpy.config.save_directory or prefix != renpy.config.save_directory:
                    continue
            data = zf.read(i)
            with open(p / filename, 'wb') as f:
                f.write(data)
    return True