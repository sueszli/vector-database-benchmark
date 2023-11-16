"""Utility script to dump information from SCons signature database."""
import getopt
import os
import sys
from dbm import whichdb
import time
import pickle
import SCons.compat
import SCons.SConsign

def my_whichdb(filename):
    if False:
        for i in range(10):
            print('nop')
    if filename[-7:] == '.dblite':
        return 'SCons.dblite'
    try:
        with open(filename + '.dblite', 'rb'):
            return 'SCons.dblite'
    except IOError:
        pass
    return whichdb(filename)

def my_import(mname):
    if False:
        while True:
            i = 10
    import imp
    if '.' in mname:
        i = mname.rfind('.')
        parent = my_import(mname[:i])
        (fp, pathname, description) = imp.find_module(mname[i + 1:], parent.__path__)
    else:
        (fp, pathname, description) = imp.find_module(mname)
    return imp.load_module(mname, fp, pathname, description)

class Flagger:
    default_value = 1

    def __setitem__(self, item, value):
        if False:
            return 10
        self.__dict__[item] = value
        self.default_value = 0

    def __getitem__(self, item):
        if False:
            return 10
        return self.__dict__.get(item, self.default_value)
Do_Call = None
Print_Directories = []
Print_Entries = []
Print_Flags = Flagger()
Verbose = 0
Readable = 0
Warns = 0

def default_mapper(entry, name):
    if False:
        while True:
            i = 10
    "\n    Stringify an entry that doesn't have an explicit mapping.\n\n    Args:\n        entry:  entry\n        name: field name\n\n    Returns: str\n\n    "
    try:
        val = eval('entry.' + name)
    except AttributeError:
        val = None
    return str(val)

def map_action(entry, _):
    if False:
        while True:
            i = 10
    '\n    Stringify an action entry and signature.\n\n    Args:\n        entry: action entry\n        second argument is not used\n\n    Returns: str\n\n    '
    try:
        bact = entry.bact
        bactsig = entry.bactsig
    except AttributeError:
        return None
    return '%s [%s]' % (bactsig, bact)

def map_timestamp(entry, _):
    if False:
        return 10
    '\n    Stringify a timestamp entry.\n\n    Args:\n        entry: timestamp entry\n        second argument is not used\n\n    Returns: str\n\n    '
    try:
        timestamp = entry.timestamp
    except AttributeError:
        timestamp = None
    if Readable and timestamp:
        return "'" + time.ctime(timestamp) + "'"
    else:
        return str(timestamp)

def map_bkids(entry, _):
    if False:
        for i in range(10):
            print('nop')
    '\n    Stringify an implicit entry.\n\n    Args:\n        entry:\n        second argument is not used\n\n    Returns: str\n\n    '
    try:
        bkids = entry.bsources + entry.bdepends + entry.bimplicit
        bkidsigs = entry.bsourcesigs + entry.bdependsigs + entry.bimplicitsigs
    except AttributeError:
        return None
    if len(bkids) != len(bkidsigs):
        global Warns
        Warns += 1
        msg = 'Warning: missing information, {} ids but {} sigs'
        result = [msg.format(len(bkids), len(bkidsigs))]
    else:
        result = []
    result += [nodeinfo_string(bkid, bkidsig, '        ') for (bkid, bkidsig) in zip(bkids, bkidsigs)]
    if not result:
        return None
    return '\n        '.join(result)
map_field = {'action': map_action, 'timestamp': map_timestamp, 'bkids': map_bkids}
map_name = {'implicit': 'bkids'}

def field(name, entry, verbose=Verbose):
    if False:
        print('Hello World!')
    if not Print_Flags[name]:
        return None
    fieldname = map_name.get(name, name)
    mapper = map_field.get(fieldname, default_mapper)
    val = mapper(entry, name)
    if verbose:
        val = name + ': ' + val
    return val

def nodeinfo_raw(name, ninfo, prefix=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    This just formats the dictionary, which we would normally use str()\n    to do, except that we want the keys sorted for deterministic output.\n    '
    d = ninfo.__getstate__()
    try:
        keys = ninfo.field_list + ['_version_id']
    except AttributeError:
        keys = sorted(d.keys())
    values = []
    for key in keys:
        values.append('%s: %s' % (repr(key), repr(d.get(key))))
    if '\n' in name:
        name = repr(name)
    return name + ': {' + ', '.join(values) + '}'

def nodeinfo_cooked(name, ninfo, prefix=''):
    if False:
        return 10
    try:
        field_list = ninfo.field_list
    except AttributeError:
        field_list = []
    if '\n' in name:
        name = repr(name)
    outlist = [name + ':'] + [f for f in [field(x, ninfo, Verbose) for x in field_list] if f]
    if Verbose:
        sep = '\n    ' + prefix
    else:
        sep = ' '
    return sep.join(outlist)
nodeinfo_string = nodeinfo_cooked

def printfield(name, entry, prefix=''):
    if False:
        while True:
            i = 10
    outlist = field('implicit', entry, 0)
    if outlist:
        if Verbose:
            print('    implicit:')
        print('        ' + outlist)
    outact = field('action', entry, 0)
    if outact:
        if Verbose:
            print('    action: ' + outact)
        else:
            print('        ' + outact)

def printentries(entries, location):
    if False:
        while True:
            i = 10
    if Print_Entries:
        for name in Print_Entries:
            try:
                entry = entries[name]
            except KeyError:
                err = "sconsign: no entry `%s' in `%s'\n" % (name, location)
                sys.stderr.write(err)
            else:
                try:
                    ninfo = entry.ninfo
                except AttributeError:
                    print(name + ':')
                else:
                    print(nodeinfo_string(name, entry.ninfo))
                printfield(name, entry.binfo)
    else:
        for name in sorted(entries.keys()):
            entry = entries[name]
            try:
                entry.ninfo
            except AttributeError:
                print(name + ':')
            else:
                print(nodeinfo_string(name, entry.ninfo))
            printfield(name, entry.binfo)

class Do_SConsignDB:

    def __init__(self, dbm_name, dbm):
        if False:
            return 10
        self.dbm_name = dbm_name
        self.dbm = dbm

    def __call__(self, fname):
        if False:
            while True:
                i = 10
        try:
            db = self.dbm.open(fname, 'r')
        except (IOError, OSError) as e:
            print_e = e
            try:
                db = self.dbm.open(os.path.splitext(fname)[0], 'r')
            except (IOError, OSError):
                try:
                    with open(fname, 'rb'):
                        pass
                except (IOError, OSError) as e:
                    print_e = e
                sys.stderr.write('sconsign: %s\n' % print_e)
                return
        except KeyboardInterrupt:
            raise
        except pickle.UnpicklingError:
            sys.stderr.write("sconsign: ignoring invalid `%s' file `%s'\n" % (self.dbm_name, fname))
            return
        except Exception as e:
            sys.stderr.write("sconsign: ignoring invalid `%s' file `%s': %s\n" % (self.dbm_name, fname, e))
            (exc_type, _, _) = sys.exc_info()
            if exc_type.__name__ == 'ValueError':
                sys.stderr.write('unrecognized pickle protocol.\n')
            return
        if Print_Directories:
            for dir in Print_Directories:
                try:
                    val = db[dir]
                except KeyError:
                    err = "sconsign: no dir `%s' in `%s'\n" % (dir, args[0])
                    sys.stderr.write(err)
                else:
                    self.printentries(dir, val)
        else:
            for dir in sorted(db.keys()):
                self.printentries(dir, db[dir])

    @staticmethod
    def printentries(dir, val):
        if False:
            i = 10
            return i + 15
        try:
            print('=== ' + dir + ':')
        except TypeError:
            print('=== ' + dir.decode() + ':')
        printentries(pickle.loads(val), dir)

def Do_SConsignDir(name):
    if False:
        while True:
            i = 10
    try:
        with open(name, 'rb') as fp:
            try:
                sconsign = SCons.SConsign.Dir(fp)
            except KeyboardInterrupt:
                raise
            except pickle.UnpicklingError:
                err = "sconsign: ignoring invalid .sconsign file `%s'\n" % name
                sys.stderr.write(err)
                return
            except Exception as e:
                err = "sconsign: ignoring invalid .sconsign file `%s': %s\n" % (name, e)
                sys.stderr.write(err)
                return
            printentries(sconsign.entries, args[0])
    except (IOError, OSError) as e:
        sys.stderr.write('sconsign: %s\n' % e)
        return

def main():
    if False:
        print('Hello World!')
    global Do_Call
    global nodeinfo_string
    global args
    global Verbose
    global Readable
    helpstr = '    Usage: sconsign [OPTIONS] [FILE ...]\n    Options:\n      -a, --act, --action         Print build action information.\n      -c, --csig                  Print content signature information.\n      -d DIR, --dir=DIR           Print only info about DIR.\n      -e ENTRY, --entry=ENTRY     Print only info about ENTRY.\n      -f FORMAT, --format=FORMAT  FILE is in the specified FORMAT.\n      -h, --help                  Print this message and exit.\n      -i, --implicit              Print implicit dependency information.\n      -r, --readable              Print timestamps in human-readable form.\n      --raw                       Print raw Python object representations.\n      -s, --size                  Print file sizes.\n      -t, --timestamp             Print timestamp information.\n      -v, --verbose               Verbose, describe each field.\n    '
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'acd:e:f:hirstv', ['act', 'action', 'csig', 'dir=', 'entry=', 'format=', 'help', 'implicit', 'raw', 'readable', 'size', 'timestamp', 'verbose'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err) + '\n')
        print(helpstr)
        sys.exit(2)
    for (o, a) in opts:
        if o in ('-a', '--act', '--action'):
            Print_Flags['action'] = 1
        elif o in ('-c', '--csig'):
            Print_Flags['csig'] = 1
        elif o in ('-d', '--dir'):
            Print_Directories.append(a)
        elif o in ('-e', '--entry'):
            Print_Entries.append(a)
        elif o in ('-f', '--format'):
            Module_Map = {'dblite': 'SCons.dblite', 'sconsign': None}
            dbm_name = Module_Map.get(a, a)
            if dbm_name:
                try:
                    if dbm_name != 'SCons.dblite':
                        dbm = my_import(dbm_name)
                    else:
                        import SCons.dblite
                        dbm = SCons.dblite
                        SCons.dblite.IGNORE_CORRUPT_DBFILES = False
                except ImportError:
                    sys.stderr.write("sconsign: illegal file format `%s'\n" % a)
                    print(helpstr)
                    sys.exit(2)
                Do_Call = Do_SConsignDB(a, dbm)
            else:
                Do_Call = Do_SConsignDir
        elif o in ('-h', '--help'):
            print(helpstr)
            sys.exit(0)
        elif o in ('-i', '--implicit'):
            Print_Flags['implicit'] = 1
        elif o in ('--raw',):
            nodeinfo_string = nodeinfo_raw
        elif o in ('-r', '--readable'):
            Readable = 1
        elif o in ('-s', '--size'):
            Print_Flags['size'] = 1
        elif o in ('-t', '--timestamp'):
            Print_Flags['timestamp'] = 1
        elif o in ('-v', '--verbose'):
            Verbose = 1
    if Do_Call:
        for a in args:
            Do_Call(a)
    else:
        if not args:
            args = ['.sconsign.dblite']
        for a in args:
            dbm_name = my_whichdb(a)
            if dbm_name:
                Map_Module = {'SCons.dblite': 'dblite'}
                if dbm_name != 'SCons.dblite':
                    dbm = my_import(dbm_name)
                else:
                    import SCons.dblite
                    dbm = SCons.dblite
                    SCons.dblite.IGNORE_CORRUPT_DBFILES = False
                Do_SConsignDB(Map_Module.get(dbm_name, dbm_name), dbm)(a)
            else:
                Do_SConsignDir(a)
        if Warns:
            print('NOTE: there were %d warnings, please check output' % Warns)
if __name__ == '__main__':
    main()
    sys.exit(0)