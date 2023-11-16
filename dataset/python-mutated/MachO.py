"""
Utilities for reading and writing Mach-O headers
"""
from __future__ import print_function
import os
import struct
import sys
from macholib.util import fileview
from .mach_o import FAT_MAGIC, FAT_MAGIC_64, LC_DYSYMTAB, LC_ID_DYLIB, LC_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_PREBOUND_DYLIB, LC_REEXPORT_DYLIB, LC_REGISTRY, LC_SEGMENT, LC_SEGMENT_64, LC_SYMTAB, MH_CIGAM, MH_CIGAM_64, MH_FILETYPE_SHORTNAMES, MH_MAGIC, MH_MAGIC_64, S_ZEROFILL, fat_arch, fat_arch64, fat_header, load_command, mach_header, mach_header_64, section, section_64
from .ptypes import sizeof
try:
    from macholib.compat import bytes
except ImportError:
    pass
try:
    unicode
except NameError:
    unicode = str
if sys.version_info[0] == 2:
    range = xrange
__all__ = ['MachO']
_RELOCATABLE = {LC_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_PREBOUND_DYLIB, LC_REEXPORT_DYLIB}
_RELOCATABLE_NAMES = {LC_LOAD_DYLIB: 'load_dylib', LC_LOAD_UPWARD_DYLIB: 'load_upward_dylib', LC_LOAD_WEAK_DYLIB: 'load_weak_dylib', LC_PREBOUND_DYLIB: 'prebound_dylib', LC_REEXPORT_DYLIB: 'reexport_dylib'}

def _shouldRelocateCommand(cmd):
    if False:
        print('Hello World!')
    '\n    Should this command id be investigated for relocation?\n    '
    return cmd in _RELOCATABLE

def lc_str_value(offset, cmd_info):
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetch the actual value of a field of type "lc_str"\n    '
    (cmd_load, cmd_cmd, cmd_data) = cmd_info
    offset -= sizeof(cmd_load) + sizeof(cmd_cmd)
    return cmd_data[offset:].strip(b'\x00')

class MachO(object):
    """
    Provides reading/writing the Mach-O header of a specific existing file.

    If allow_unknown_load_commands is True, allows unknown load commands.
    Otherwise, raises ValueError if the file contains an unknown load command.
    """

    def __init__(self, filename, allow_unknown_load_commands=False):
        if False:
            while True:
                i = 10
        self.graphident = filename
        self.filename = filename
        self.loader_path = os.path.dirname(filename)
        self.fat = None
        self.headers = []
        self.allow_unknown_load_commands = allow_unknown_load_commands
        with open(filename, 'rb') as fp:
            self.load(fp)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<MachO filename=%r>' % (self.filename,)

    def load(self, fh):
        if False:
            print('Hello World!')
        assert fh.tell() == 0
        header = struct.unpack('>I', fh.read(4))[0]
        fh.seek(0)
        if header in (FAT_MAGIC, FAT_MAGIC_64):
            self.load_fat(fh)
        else:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(0)
            self.load_header(fh, 0, size)

    def load_fat(self, fh):
        if False:
            i = 10
            return i + 15
        self.fat = fat_header.from_fileobj(fh)
        if self.fat.magic == FAT_MAGIC:
            archs = [fat_arch.from_fileobj(fh) for i in range(self.fat.nfat_arch)]
        elif self.fat.magic == FAT_MAGIC_64:
            archs = [fat_arch64.from_fileobj(fh) for i in range(self.fat.nfat_arch)]
        else:
            raise ValueError('Unknown fat header magic: %r' % self.fat.magic)
        for arch in archs:
            self.load_header(fh, arch.offset, arch.size)

    def rewriteLoadCommands(self, *args, **kw):
        if False:
            while True:
                i = 10
        changed = False
        for header in self.headers:
            if header.rewriteLoadCommands(*args, **kw):
                changed = True
        return changed

    def load_header(self, fh, offset, size):
        if False:
            while True:
                i = 10
        fh.seek(offset)
        header = struct.unpack('>I', fh.read(4))[0]
        fh.seek(offset)
        if header == MH_MAGIC:
            (magic, hdr, endian) = (MH_MAGIC, mach_header, '>')
        elif header == MH_CIGAM:
            (magic, hdr, endian) = (MH_CIGAM, mach_header, '<')
        elif header == MH_MAGIC_64:
            (magic, hdr, endian) = (MH_MAGIC_64, mach_header_64, '>')
        elif header == MH_CIGAM_64:
            (magic, hdr, endian) = (MH_CIGAM_64, mach_header_64, '<')
        else:
            raise ValueError('Unknown Mach-O header: 0x%08x in %r' % (header, fh))
        hdr = MachOHeader(self, fh, offset, size, magic, hdr, endian, self.allow_unknown_load_commands)
        self.headers.append(hdr)

    def write(self, f):
        if False:
            for i in range(10):
                print('nop')
        for header in self.headers:
            header.write(f)

class MachOHeader(object):
    """
    Provides reading/writing the Mach-O header of a specific existing file.

    If allow_unknown_load_commands is True, allows unknown load commands.
    Otherwise, raises ValueError if the file contains an unknown load command.
    """

    def __init__(self, parent, fh, offset, size, magic, hdr, endian, allow_unknown_load_commands=False):
        if False:
            return 10
        self.MH_MAGIC = magic
        self.mach_header = hdr
        self.parent = parent
        self.offset = offset
        self.size = size
        self.endian = endian
        self.header = None
        self.commands = None
        self.id_cmd = None
        self.sizediff = None
        self.total_size = None
        self.low_offset = None
        self.filetype = None
        self.headers = []
        self.allow_unknown_load_commands = allow_unknown_load_commands
        self.load(fh)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s filename=%r offset=%d size=%d endian=%r>' % (type(self).__name__, self.parent.filename, self.offset, self.size, self.endian)

    def load(self, fh):
        if False:
            print('Hello World!')
        fh = fileview(fh, self.offset, self.size)
        fh.seek(0)
        self.sizediff = 0
        kw = {'_endian_': self.endian}
        header = self.mach_header.from_fileobj(fh, **kw)
        self.header = header
        cmd = self.commands = []
        self.filetype = self.get_filetype_shortname(header.filetype)
        read_bytes = 0
        low_offset = sys.maxsize
        for i in range(header.ncmds):
            cmd_load = load_command.from_fileobj(fh, **kw)
            klass = LC_REGISTRY.get(cmd_load.cmd, None)
            if klass is None:
                if not self.allow_unknown_load_commands:
                    raise ValueError('Unknown load command: %d' % (cmd_load.cmd,))
                data_size = cmd_load.cmdsize - sizeof(load_command)
                cmd_data = fh.read(data_size)
                cmd.append((cmd_load, cmd_load, cmd_data))
                read_bytes += cmd_load.cmdsize
                continue
            cmd_cmd = klass.from_fileobj(fh, **kw)
            if cmd_load.cmd == LC_ID_DYLIB:
                if self.id_cmd is not None:
                    raise ValueError('This dylib already has an id')
                self.id_cmd = i
            if cmd_load.cmd in (LC_SEGMENT, LC_SEGMENT_64):
                segs = []
                if cmd_load.cmd == LC_SEGMENT:
                    section_cls = section
                else:
                    section_cls = section_64
                expected_size = sizeof(klass) + sizeof(load_command) + sizeof(section_cls) * cmd_cmd.nsects
                if cmd_load.cmdsize != expected_size:
                    raise ValueError('Segment size mismatch')
                if cmd_cmd.nsects == 0:
                    if cmd_cmd.filesize != 0:
                        low_offset = min(low_offset, cmd_cmd.fileoff)
                else:
                    for _j in range(cmd_cmd.nsects):
                        seg = section_cls.from_fileobj(fh, **kw)
                        not_zerofill = seg.flags & S_ZEROFILL != S_ZEROFILL
                        if seg.offset > 0 and seg.size > 0 and not_zerofill:
                            low_offset = min(low_offset, seg.offset)
                        if not_zerofill:
                            c = fh.tell()
                            fh.seek(seg.offset)
                            sd = fh.read(seg.size)
                            seg.add_section_data(sd)
                            fh.seek(c)
                        segs.append(seg)
                cmd_data = segs
            else:
                data_size = cmd_load.cmdsize - sizeof(klass) - sizeof(load_command)
                cmd_data = fh.read(data_size)
            cmd.append((cmd_load, cmd_cmd, cmd_data))
            read_bytes += cmd_load.cmdsize
        if read_bytes != header.sizeofcmds:
            raise ValueError('Read %d bytes, header reports %d bytes' % (read_bytes, header.sizeofcmds))
        self.total_size = sizeof(self.mach_header) + read_bytes
        self.low_offset = low_offset

    def walkRelocatables(self, shouldRelocateCommand=_shouldRelocateCommand):
        if False:
            for i in range(10):
                print('nop')
        '\n        for all relocatable commands\n        yield (command_index, command_name, filename)\n        '
        for (idx, (lc, cmd, data)) in enumerate(self.commands):
            if shouldRelocateCommand(lc.cmd):
                name = _RELOCATABLE_NAMES[lc.cmd]
                ofs = cmd.name - sizeof(lc.__class__) - sizeof(cmd.__class__)
                yield (idx, name, data[ofs:data.find(b'\x00', ofs)].decode(sys.getfilesystemencoding()))

    def rewriteInstallNameCommand(self, loadcmd):
        if False:
            for i in range(10):
                print('nop')
        'Rewrite the load command of this dylib'
        if self.id_cmd is not None:
            self.rewriteDataForCommand(self.id_cmd, loadcmd)
            return True
        return False

    def changedHeaderSizeBy(self, bytes):
        if False:
            while True:
                i = 10
        self.sizediff += bytes
        if self.total_size + self.sizediff > self.low_offset:
            print('WARNING: Mach-O header in %r may be too large to relocate' % (self.parent.filename,))

    def rewriteLoadCommands(self, changefunc):
        if False:
            for i in range(10):
                print('nop')
        '\n        Rewrite the load commands based upon a change dictionary\n        '
        data = changefunc(self.parent.filename)
        changed = False
        if data is not None:
            if self.rewriteInstallNameCommand(data.encode(sys.getfilesystemencoding())):
                changed = True
        for (idx, _name, filename) in self.walkRelocatables():
            data = changefunc(filename)
            if data is not None:
                if self.rewriteDataForCommand(idx, data.encode(sys.getfilesystemencoding())):
                    changed = True
        return changed

    def rewriteDataForCommand(self, idx, data):
        if False:
            while True:
                i = 10
        (lc, cmd, old_data) = self.commands[idx]
        hdrsize = sizeof(lc.__class__) + sizeof(cmd.__class__)
        align = struct.calcsize('Q')
        data = data + b'\x00' * (align - len(data) % align)
        newsize = hdrsize + len(data)
        self.commands[idx] = (lc, cmd, data)
        self.changedHeaderSizeBy(newsize - lc.cmdsize)
        (lc.cmdsize, cmd.name) = (newsize, hdrsize)
        return True

    def synchronize_size(self):
        if False:
            i = 10
            return i + 15
        if self.total_size + self.sizediff > self.low_offset:
            raise ValueError('New Mach-O header is too large to relocate in %r (new size=%r, max size=%r, delta=%r)' % (self.parent.filename, self.total_size + self.sizediff, self.low_offset, self.sizediff))
        self.header.sizeofcmds += self.sizediff
        self.total_size = sizeof(self.mach_header) + self.header.sizeofcmds
        self.sizediff = 0

    def write(self, fileobj):
        if False:
            return 10
        fileobj = fileview(fileobj, self.offset, self.size)
        fileobj.seek(0)
        self.synchronize_size()
        self.header.to_fileobj(fileobj)
        for (lc, cmd, data) in self.commands:
            lc.to_fileobj(fileobj)
            cmd.to_fileobj(fileobj)
            if sys.version_info[0] == 2:
                if isinstance(data, unicode):
                    fileobj.write(data.encode(sys.getfilesystemencoding()))
                elif isinstance(data, (bytes, str)):
                    fileobj.write(data)
                else:
                    for obj in data:
                        obj.to_fileobj(fileobj)
            elif isinstance(data, str):
                fileobj.write(data.encode(sys.getfilesystemencoding()))
            elif isinstance(data, bytes):
                fileobj.write(data)
            else:
                for obj in data:
                    obj.to_fileobj(fileobj)
        fileobj.write(b'\x00' * (self.low_offset - fileobj.tell()))

    def getSymbolTableCommand(self):
        if False:
            return 10
        for (lc, cmd, _data) in self.commands:
            if lc.cmd == LC_SYMTAB:
                return cmd
        return None

    def getDynamicSymbolTableCommand(self):
        if False:
            for i in range(10):
                print('nop')
        for (lc, cmd, _data) in self.commands:
            if lc.cmd == LC_DYSYMTAB:
                return cmd
        return None

    def get_filetype_shortname(self, filetype):
        if False:
            while True:
                i = 10
        if filetype in MH_FILETYPE_SHORTNAMES:
            return MH_FILETYPE_SHORTNAMES[filetype]
        else:
            return 'unknown'

def main(fn):
    if False:
        print('Hello World!')
    m = MachO(fn)
    seen = set()
    for header in m.headers:
        for (_idx, name, other) in header.walkRelocatables():
            if other not in seen:
                seen.add(other)
                print('\t' + name + ': ' + other)
if __name__ == '__main__':
    import sys
    files = sys.argv[1:] or ['/bin/ls']
    for fn in files:
        print(fn)
        main(fn)