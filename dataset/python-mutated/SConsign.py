"""SCons.SConsign

Writing and reading information to the .sconsign file or files.

"""
from __future__ import print_function
__revision__ = 'src/engine/SCons/SConsign.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.compat
import os
import pickle
import SCons.dblite
import SCons.Warnings
from SCons.compat import PICKLE_PROTOCOL

def corrupt_dblite_warning(filename):
    if False:
        while True:
            i = 10
    SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning, 'Ignoring corrupt .sconsign file: %s' % filename)
SCons.dblite.ignore_corrupt_dbfiles = 1
SCons.dblite.corruption_warning = corrupt_dblite_warning
sig_files = []
DataBase = {}
DB_Module = SCons.dblite
DB_Name = '.sconsign'
DB_sync_list = []

def Get_DataBase(dir):
    if False:
        return 10
    global DataBase, DB_Module, DB_Name
    top = dir.fs.Top
    if not os.path.isabs(DB_Name) and top.repositories:
        mode = 'c'
        for d in [top] + top.repositories:
            if dir.is_under(d):
                try:
                    return (DataBase[d], mode)
                except KeyError:
                    path = d.entry_abspath(DB_Name)
                    try:
                        db = DataBase[d] = DB_Module.open(path, mode)
                    except (IOError, OSError):
                        pass
                    else:
                        if mode != 'r':
                            DB_sync_list.append(db)
                        return (db, mode)
            mode = 'r'
    try:
        return (DataBase[top], 'c')
    except KeyError:
        db = DataBase[top] = DB_Module.open(DB_Name, 'c')
        DB_sync_list.append(db)
        return (db, 'c')
    except TypeError:
        print('DataBase =', DataBase)
        raise

def Reset():
    if False:
        print('Hello World!')
    'Reset global state.  Used by unit tests that end up using\n    SConsign multiple times to get a clean slate for each test.'
    global sig_files, DB_sync_list
    sig_files = []
    DB_sync_list = []
normcase = os.path.normcase

def write():
    if False:
        for i in range(10):
            print('nop')
    global sig_files
    for sig_file in sig_files:
        sig_file.write(sync=0)
    for db in DB_sync_list:
        try:
            syncmethod = db.sync
        except AttributeError:
            pass
        else:
            syncmethod()
        try:
            closemethod = db.close
        except AttributeError:
            pass
        else:
            closemethod()

class SConsignEntry(object):
    """
    Wrapper class for the generic entry in a .sconsign file.
    The Node subclass populates it with attributes as it pleases.

    XXX As coded below, we do expect a '.binfo' attribute to be added,
    but we'll probably generalize this in the next refactorings.
    """
    __slots__ = ('binfo', 'ninfo', '__weakref__')
    current_version_id = 2

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def convert_to_sconsign(self):
        if False:
            for i in range(10):
                print('nop')
        self.binfo.convert_to_sconsign()

    def convert_from_sconsign(self, dir, name):
        if False:
            print('Hello World!')
        self.binfo.convert_from_sconsign(dir, name)

    def __getstate__(self):
        if False:
            return 10
        state = getattr(self, '__dict__', {}).copy()
        for obj in type(self).mro():
            for name in getattr(obj, '__slots__', ()):
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        state['_version_id'] = self.current_version_id
        try:
            del state['__weakref__']
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in state.items():
            if key not in ('_version_id', '__weakref__'):
                setattr(self, key, value)

class Base(object):
    """
    This is the controlling class for the signatures for the collection of
    entries associated with a specific directory.  The actual directory
    association will be maintained by a subclass that is specific to
    the underlying storage method.  This class provides a common set of
    methods for fetching and storing the individual bits of information
    that make up signature entry.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.entries = {}
        self.dirty = False
        self.to_be_merged = {}

    def get_entry(self, filename):
        if False:
            print('Hello World!')
        '\n        Fetch the specified entry attribute.\n        '
        return self.entries[filename]

    def set_entry(self, filename, obj):
        if False:
            i = 10
            return i + 15
        '\n        Set the entry.\n        '
        self.entries[filename] = obj
        self.dirty = True

    def do_not_set_entry(self, filename, obj):
        if False:
            i = 10
            return i + 15
        pass

    def store_info(self, filename, node):
        if False:
            return 10
        entry = node.get_stored_info()
        entry.binfo.merge(node.get_binfo())
        self.to_be_merged[filename] = node
        self.dirty = True

    def do_not_store_info(self, filename, node):
        if False:
            while True:
                i = 10
        pass

    def merge(self):
        if False:
            while True:
                i = 10
        for (key, node) in self.to_be_merged.items():
            entry = node.get_stored_info()
            try:
                ninfo = entry.ninfo
            except AttributeError:
                pass
            else:
                ninfo.merge(node.get_ninfo())
            self.entries[key] = entry
        self.to_be_merged = {}

class DB(Base):
    """
    A Base subclass that reads and writes signature information
    from a global .sconsign.db* file--the actual file suffix is
    determined by the database module.
    """

    def __init__(self, dir):
        if False:
            print('Hello World!')
        Base.__init__(self)
        self.dir = dir
        (db, mode) = Get_DataBase(dir)
        path = normcase(dir.get_tpath())
        try:
            rawentries = db[path]
        except KeyError:
            pass
        else:
            try:
                self.entries = pickle.loads(rawentries)
                if not isinstance(self.entries, dict):
                    self.entries = {}
                    raise TypeError
            except KeyboardInterrupt:
                raise
            except Exception as e:
                SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning, 'Ignoring corrupt sconsign entry : %s (%s)\n' % (self.dir.get_tpath(), e))
            for (key, entry) in self.entries.items():
                entry.convert_from_sconsign(dir, key)
        if mode == 'r':
            self.set_entry = self.do_not_set_entry
            self.store_info = self.do_not_store_info
        global sig_files
        sig_files.append(self)

    def write(self, sync=1):
        if False:
            return 10
        if not self.dirty:
            return
        self.merge()
        (db, mode) = Get_DataBase(self.dir)
        path = normcase(self.dir.get_internal_path())
        for (key, entry) in self.entries.items():
            entry.convert_to_sconsign()
        db[path] = pickle.dumps(self.entries, PICKLE_PROTOCOL)
        if sync:
            try:
                syncmethod = db.sync
            except AttributeError:
                pass
            else:
                syncmethod()

class Dir(Base):

    def __init__(self, fp=None, dir=None):
        if False:
            print('Hello World!')
        '\n        fp - file pointer to read entries from\n        '
        Base.__init__(self)
        if not fp:
            return
        self.entries = pickle.load(fp)
        if not isinstance(self.entries, dict):
            self.entries = {}
            raise TypeError
        if dir:
            for (key, entry) in self.entries.items():
                entry.convert_from_sconsign(dir, key)

class DirFile(Dir):
    """
    Encapsulates reading and writing a per-directory .sconsign file.
    """

    def __init__(self, dir):
        if False:
            i = 10
            return i + 15
        '\n        dir - the directory for the file\n        '
        self.dir = dir
        self.sconsign = os.path.join(dir.get_internal_path(), '.sconsign')
        try:
            fp = open(self.sconsign, 'rb')
        except IOError:
            fp = None
        try:
            Dir.__init__(self, fp, dir)
        except KeyboardInterrupt:
            raise
        except Exception:
            SCons.Warnings.warn(SCons.Warnings.CorruptSConsignWarning, 'Ignoring corrupt .sconsign file: %s' % self.sconsign)
        try:
            fp.close()
        except AttributeError:
            pass
        global sig_files
        sig_files.append(self)

    def write(self, sync=1):
        if False:
            i = 10
            return i + 15
        "\n        Write the .sconsign file to disk.\n\n        Try to write to a temporary file first, and rename it if we\n        succeed.  If we can't write to the temporary file, it's\n        probably because the directory isn't writable (and if so,\n        how did we build anything in this directory, anyway?), so\n        try to write directly to the .sconsign file as a backup.\n        If we can't rename, try to copy the temporary contents back\n        to the .sconsign file.  Either way, always try to remove\n        the temporary file at the end.\n        "
        if not self.dirty:
            return
        self.merge()
        temp = os.path.join(self.dir.get_internal_path(), '.scons%d' % os.getpid())
        try:
            file = open(temp, 'wb')
            fname = temp
        except IOError:
            try:
                file = open(self.sconsign, 'wb')
                fname = self.sconsign
            except IOError:
                return
        for (key, entry) in self.entries.items():
            entry.convert_to_sconsign()
        pickle.dump(self.entries, file, PICKLE_PROTOCOL)
        file.close()
        if fname != self.sconsign:
            try:
                mode = os.stat(self.sconsign)[0]
                os.chmod(self.sconsign, 438)
                os.unlink(self.sconsign)
            except (IOError, OSError):
                pass
            try:
                os.rename(fname, self.sconsign)
            except OSError:
                with open(self.sconsign, 'wb') as f, open(fname, 'rb') as f2:
                    f.write(f2.read())
                os.chmod(self.sconsign, mode)
        try:
            os.unlink(temp)
        except (IOError, OSError):
            pass
ForDirectory = DB

def File(name, dbm_module=None):
    if False:
        return 10
    '\n    Arrange for all signatures to be stored in a global .sconsign.db*\n    file.\n    '
    global ForDirectory, DB_Name, DB_Module
    if name is None:
        ForDirectory = DirFile
        DB_Module = None
    else:
        ForDirectory = DB
        DB_Name = name
        if dbm_module is not None:
            DB_Module = dbm_module