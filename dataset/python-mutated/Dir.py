__revision__ = 'src/engine/SCons/Scanner/Dir.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Node.FS
import SCons.Scanner

def only_dirs(nodes):
    if False:
        i = 10
        return i + 15
    is_Dir = lambda n: isinstance(n.disambiguate(), SCons.Node.FS.Dir)
    return [node for node in nodes if is_Dir(node)]

def DirScanner(**kw):
    if False:
        return 10
    'Return a prototype Scanner instance for scanning\n    directories for on-disk files'
    kw['node_factory'] = SCons.Node.FS.Entry
    kw['recursive'] = only_dirs
    return SCons.Scanner.Base(scan_on_disk, 'DirScanner', **kw)

def DirEntryScanner(**kw):
    if False:
        while True:
            i = 10
    'Return a prototype Scanner instance for "scanning"\n    directory Nodes for their in-memory entries'
    kw['node_factory'] = SCons.Node.FS.Entry
    kw['recursive'] = None
    return SCons.Scanner.Base(scan_in_memory, 'DirEntryScanner', **kw)
skip_entry = {}
skip_entry_list = ['.', '..', '.sconsign', '.sconsign.dblite', '.sconsign.dir', '.sconsign.pag', '.sconsign.dat', '.sconsign.bak', '.sconsign.db']
for skip in skip_entry_list:
    skip_entry[skip] = 1
    skip_entry[SCons.Node.FS._my_normcase(skip)] = 1
do_not_scan = lambda k: k not in skip_entry

def scan_on_disk(node, env, path=()):
    if False:
        return 10
    '\n    Scans a directory for on-disk files and directories therein.\n\n    Looking up the entries will add these to the in-memory Node tree\n    representation of the file system, so all we have to do is just\n    that and then call the in-memory scanning function.\n    '
    try:
        flist = node.fs.listdir(node.get_abspath())
    except (IOError, OSError):
        return []
    e = node.Entry
    for f in filter(do_not_scan, flist):
        e('./' + f)
    return scan_in_memory(node, env, path)

def scan_in_memory(node, env, path=()):
    if False:
        print('Hello World!')
    '\n    "Scans" a Node.FS.Dir for its in-memory entries.\n    '
    try:
        entries = node.entries
    except AttributeError:
        return []
    entry_list = sorted(filter(do_not_scan, list(entries.keys())))
    return [entries[n] for n in entry_list]