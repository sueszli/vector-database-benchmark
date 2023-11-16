from volatility import renderers
import volatility.obj as obj
import volatility.poolscan as poolscan
import volatility.plugins.common as common
import volatility.plugins.gui.windowstations as windowstations
from volatility.renderers.basic import Hex, Address

class PoolScanAtom(poolscan.PoolScanner):
    """Pool scanner for atom tables"""

    def __init__(self, address_space):
        if False:
            while True:
                i = 10
        poolscan.PoolScanner.__init__(self, address_space)
        self.pooltag = 'AtmT'
        self.struct_name = '_RTL_ATOM_TABLE'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 200)), ('CheckPoolType', dict(paged=True, non_paged=True, free=True))]
        profile = self.address_space.profile
        build = (profile.metadata.get('major', 0), profile.metadata.get('minor', 0))
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            fixup = 8 if build > (5, 1) else 0
        else:
            fixup = 16 if build > (5, 1) else 0
        self.padding = fixup

class AtomScan(common.AbstractScanCommand):
    """Pool scanner for atom tables"""
    scanners = [PoolScanAtom]

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        common.AbstractScanCommand.__init__(self, config, *args, **kwargs)
        config.add_option('SORT-BY', short_option='s', type='choice', choices=['atom', 'refcount', 'offset'], default='offset', help='Sort by [offset | atom | refcount]', action='store')
    text_sort_column = 'Atom'

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [(self.offset_column(), '[addr]'), ('AtomOfs(V)', '[addrpad]'), ('Atom', '[addr]'), ('Refs', '6'), ('Pinned', '6'), ('Name', '')])
        for atom_table in data:
            atoms = [a for a in atom_table.atoms() if a.is_string_atom()]
            if self._config.SORT_BY == 'atom':
                attr = 'Atom'
            elif self._config.SORT_BY == 'refcount':
                attr = 'ReferenceCount'
            else:
                attr = 'obj_offset'
            for atom in sorted(atoms, key=lambda x: getattr(x, attr)):
                self.table_row(outfd, atom_table.obj_offset, atom.obj_offset, atom.Atom, atom.ReferenceCount, atom.Pinned, str(atom.Name or ''))

    def unified_output(self, data):
        if False:
            return 10
        return renderers.TreeGrid([(self.offset_column(), Address), ('AtomOfs(V)', Address), ('Atom', Hex), ('Refs', int), ('Pinned', int), ('Name', str)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for atom_table in data:
            atoms = [a for a in atom_table.atoms() if a.is_string_atom()]
            if self._config.SORT_BY == 'atom':
                attr = 'Atom'
            elif self._config.SORT_BY == 'refcount':
                attr = 'ReferenceCount'
            else:
                attr = 'obj_offset'
            for atom in sorted(atoms, key=lambda x: getattr(x, attr)):
                yield (0, [Address(atom_table.obj_offset), Address(atom.obj_offset), Hex(atom.Atom), int(atom.ReferenceCount), int(atom.Pinned), str(atom.Name or '')])

class Atoms(common.AbstractWindowsCommand):
    """Print session and window station atom tables"""

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        seen = []
        for wndsta in windowstations.WndScan(self._config).calculate():
            offset = wndsta.obj_native_vm.vtop(wndsta.pGlobalAtomTable)
            if offset in seen:
                continue
            seen.append(offset)
            atom_table = wndsta.AtomTable
            if atom_table.is_valid():
                yield (atom_table, wndsta)
        for table in AtomScan(self._config).calculate():
            if table.PhysicalAddress not in seen:
                yield (table, obj.NoneObject('No windowstation'))
    text_sort_column = 'Atom'

    def unified_output(self, data):
        if False:
            return 10
        return renderers.TreeGrid([('Offset(V)', Address), ('Session', int), ('WindowStation', str), ('Atom', Hex), ('RefCount', int), ('HIndex', int), ('Pinned', int), ('Name', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for (atom_table, window_station) in data:
            for atom in atom_table.atoms():
                if not atom.is_string_atom():
                    continue
                yield (0, [Address(atom_table.PhysicalAddress), int(window_station.dwSessionId), str(window_station.Name or ''), Hex(atom.Atom), int(atom.ReferenceCount), int(atom.HandleIndex), int(atom.Pinned), str(atom.Name or '')])

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Offset(V)', '[addr]'), ('Session', '^10'), ('WindowStation', '^18'), ('Atom', '[addr]'), ('RefCount', '^10'), ('HIndex', '^10'), ('Pinned', '^10'), ('Name', '')])
        for (atom_table, window_station) in data:
            for atom in atom_table.atoms():
                if not atom.is_string_atom():
                    continue
                self.table_row(outfd, atom_table.PhysicalAddress, window_station.dwSessionId, window_station.Name, atom.Atom, atom.ReferenceCount, atom.HandleIndex, atom.Pinned, str(atom.Name or ''))