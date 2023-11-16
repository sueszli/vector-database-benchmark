import os
from collections import deque
from macholib.dyld import framework_info
from macholib.MachOGraph import MachOGraph, MissingMachO
from macholib.util import flipwritable, has_filename_filter, in_system_path, iter_platform_files, mergecopy, mergetree

class ExcludedMachO(MissingMachO):
    pass

class FilteredMachOGraph(MachOGraph):

    def __init__(self, delegate, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(FilteredMachOGraph, self).__init__(*args, **kwargs)
        self.delegate = delegate

    def createNode(self, cls, name):
        if False:
            print('Hello World!')
        cls = self.delegate.getClass(name, cls)
        res = super(FilteredMachOGraph, self).createNode(cls, name)
        return self.delegate.update_node(res)

    def locate(self, filename, loader=None):
        if False:
            while True:
                i = 10
        newname = super(FilteredMachOGraph, self).locate(filename, loader)
        if newname is None:
            return None
        return self.delegate.locate(newname, loader=loader)

class MachOStandalone(object):

    def __init__(self, base, dest=None, graph=None, env=None, executable_path=None):
        if False:
            for i in range(10):
                print('nop')
        self.base = os.path.join(os.path.abspath(base), '')
        if dest is None:
            dest = os.path.join(self.base, 'Contents', 'Frameworks')
        self.dest = dest
        self.mm = FilteredMachOGraph(self, graph=graph, env=env, executable_path=executable_path)
        self.changemap = {}
        self.excludes = []
        self.pending = deque()

    def update_node(self, m):
        if False:
            print('Hello World!')
        return m

    def getClass(self, name, cls):
        if False:
            i = 10
            return i + 15
        if in_system_path(name):
            return ExcludedMachO
        for base in self.excludes:
            if name.startswith(base):
                return ExcludedMachO
        return cls

    def locate(self, filename, loader=None):
        if False:
            i = 10
            return i + 15
        if in_system_path(filename):
            return filename
        if filename.startswith(self.base):
            return filename
        for base in self.excludes:
            if filename.startswith(base):
                return filename
        if filename in self.changemap:
            return self.changemap[filename]
        info = framework_info(filename)
        if info is None:
            res = self.copy_dylib(filename)
            self.changemap[filename] = res
            return res
        else:
            res = self.copy_framework(info)
            self.changemap[filename] = res
            return res

    def copy_dylib(self, filename):
        if False:
            for i in range(10):
                print('nop')
        if os.path.islink(filename):
            dest = os.path.join(self.dest, os.path.basename(os.path.realpath(filename)))
        else:
            dest = os.path.join(self.dest, os.path.basename(filename))
        if not os.path.exists(dest):
            self.mergecopy(filename, dest)
        return dest

    def mergecopy(self, src, dest):
        if False:
            return 10
        return mergecopy(src, dest)

    def mergetree(self, src, dest):
        if False:
            return 10
        return mergetree(src, dest)

    def copy_framework(self, info):
        if False:
            i = 10
            return i + 15
        dest = os.path.join(self.dest, info['shortname'] + '.framework')
        destfn = os.path.join(self.dest, info['name'])
        src = os.path.join(info['location'], info['shortname'] + '.framework')
        if not os.path.exists(dest):
            self.mergetree(src, dest)
            self.pending.append((destfn, iter_platform_files(dest)))
        return destfn

    def run(self, platfiles=None, contents=None):
        if False:
            for i in range(10):
                print('nop')
        mm = self.mm
        if contents is None:
            contents = '@executable_path/..'
        if platfiles is None:
            platfiles = iter_platform_files(self.base)
        for fn in platfiles:
            mm.run_file(fn)
        while self.pending:
            (fmwk, files) = self.pending.popleft()
            ref = mm.findNode(fmwk)
            for fn in files:
                mm.run_file(fn, caller=ref)
        changemap = {}
        skipcontents = os.path.join(os.path.dirname(self.dest), '')
        machfiles = []
        for node in mm.flatten(has_filename_filter):
            machfiles.append(node)
            dest = os.path.join(contents, os.path.normpath(node.filename[len(skipcontents):]))
            changemap[node.filename] = dest

        def changefunc(path):
            if False:
                print('Hello World!')
            if path.startswith('@loader_path/'):
                return path
            res = mm.locate(path)
            rv = changemap.get(res)
            if rv is None and path.startswith('@loader_path/'):
                rv = changemap.get(mm.locate(mm.trans_table.get((node.filename, path))))
            return rv
        for node in machfiles:
            fn = mm.locate(node.filename)
            if fn is None:
                continue
            rewroteAny = False
            for _header in node.headers:
                if node.rewriteLoadCommands(changefunc):
                    rewroteAny = True
            if rewroteAny:
                old_mode = flipwritable(fn)
                try:
                    with open(fn, 'rb+') as f:
                        for _header in node.headers:
                            f.seek(0)
                            node.write(f)
                        f.seek(0, 2)
                        f.flush()
                finally:
                    flipwritable(fn, old_mode)
        allfiles = [mm.locate(node.filename) for node in machfiles]
        return set(filter(None, allfiles))