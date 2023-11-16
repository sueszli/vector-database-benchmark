import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.gui.atoms as atoms
import volatility.plugins.gui.constants as consts
import volatility.plugins.gui.sessions as sessions
message_offsets_x86 = [(1749152, 1749088), (1748640, 1748576), (1754688, 1754624), (1741824, 1741760), (1741344, 1741280), (1732352, 1732288), (1704760, 1704896), (1791488, 1791424), (1774656, 1774592), (1958112, 1958048), (1966560, 1966496), (1962208, 1962144), (2201600, 2201536), (2222080, 2222016), (2205952, 2205888)]
message_offsets_x64 = [(3881088, 3881024), (3885184, 3885120), (2669088, 2669024), (2656800, 2656736), (2661408, 2661344), (2991232, 2991168), (2995872, 2995808), (3016864, 3016800), (3016352, 3016288)]

class MessageHooks(atoms.Atoms, sessions.SessionsMixin):
    """List desktop and thread window message hooks"""

    def calculate(self):
        if False:
            while True:
                i = 10
        atom_tables = dict(((atom_table, winsta) for (atom_table, winsta) in atoms.Atoms(self._config).calculate()))
        window_stations = [winsta for winsta in atom_tables.values() if winsta]
        for winsta in window_stations:
            yield (winsta, atom_tables)

    def translate_atom(self, winsta, atom_tables, atom_id):
        if False:
            i = 10
            return i + 15
        '\n        Translate an atom into an atom name.\n\n        @param winsta: a tagWINDOWSTATION in the proper \n        session space \n\n        @param atom_tables: a dictionary with _RTL_ATOM_TABLE\n        instances as the keys and owning window stations as\n        the values. \n\n        @param index: the index into the atom handle table. \n        '
        if consts.DEFAULT_ATOMS.has_key(atom_id):
            return consts.DEFAULT_ATOMS[atom_id].Name
        table_list = [table for (table, window_station) in atom_tables.items() if window_station == None]
        table_list.append(winsta.AtomTable)
        for table in table_list:
            atom = table.find_atom(atom_id)
            if atom:
                return atom.Name
        return obj.NoneObject('Cannot translate atom {0:#x}'.format(atom_id))

    def translate_hmod(self, winsta, atom_tables, index):
        if False:
            while True:
                i = 10
        "\n        Translate an ihmod (index into a handle table) into\n        an atom. This requires locating the win32k!_aatomSysLoaded \n        symbol. If the  symbol cannot be found, we'll just report \n        back the ihmod value. \n\n        @param winsta: a tagWINDOWSTATION in the proper \n        session space \n\n        @param atom_tables: a dictionary with _RTL_ATOM_TABLE\n        instances as the keys and owning window stations as\n        the values. \n\n        @param index: the index into the atom handle table. \n        "
        if index == -1:
            return '(Current Module)'
        kernel_space = utils.load_as(self._config)
        session = self.find_session_space(kernel_space, winsta.dwSessionId)
        if not session:
            return hex(index)
        if winsta.obj_vm.profile.metadata.get('memory_model', '32bit') == '32bit':
            message_offsets = message_offsets_x86
        else:
            message_offsets = message_offsets_x64
        for (count_offset, table_offset) in message_offsets:
            count = obj.Object('unsigned long', offset=session.Win32KBase + count_offset, vm=session.obj_vm)
            if count == None or count == 0 or count > 32 or (count <= index):
                continue
            atomlist = obj.Object('Array', targetType='unsigned short', offset=session.Win32KBase + table_offset, count=count, vm=session.obj_vm)
            atom_id = atomlist[index]
            module = self.translate_atom(winsta, atom_tables, atom_id)
            if module:
                return module
        return hex(index)

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        'Render output in table form'
        self.table_header(outfd, [('Offset(V)', '[addrpad]'), ('Sess', '<6'), ('Desktop', '20'), ('Thread', '30'), ('Filter', '20'), ('Flags', '20'), ('Function', '[addrpad]'), ('Module', '')])
        for (winsta, atom_tables) in data:
            for desk in winsta.desktops():
                for (name, hook) in desk.hooks():
                    module = self.translate_hmod(winsta, atom_tables, hook.ihmod)
                    self.table_row(outfd, hook.obj_offset, winsta.dwSessionId, '{0}\\{1}'.format(winsta.Name, desk.Name), '<any>', name, str(hook.flags), hook.offPfn, module)
                for thrd in desk.threads():
                    info = '{0} ({1} {2})'.format(thrd.pEThread.Cid.UniqueThread, thrd.ppi.Process.ImageFileName, thrd.ppi.Process.UniqueProcessId)
                    for (name, hook) in thrd.hooks():
                        module = self.translate_hmod(winsta, atom_tables, hook.ihmod)
                        self.table_row(outfd, hook.obj_offset, winsta.dwSessionId, '{0}\\{1}'.format(winsta.Name, desk.Name), info, name, str(hook.flags), hook.offPfn, module)

    def render_block(self, outfd, data):
        if False:
            while True:
                i = 10
        'Render output as a block'

        def write_block(outfd, winsta, desk, hook, module, thread):
            if False:
                while True:
                    i = 10
            outfd.write('{0:<10} : {1:#x}\n'.format('Offset(V)', hook.obj_offset))
            outfd.write('{0:<10} : {1}\n'.format('Session', winsta.dwSessionId))
            outfd.write('{0:<10} : {1}\n'.format('Desktop', '{0}\\{1}'.format(winsta.Name, desk.Name)))
            outfd.write('{0:<10} : {1}\n'.format('Thread', thread))
            outfd.write('{0:<10} : {1}\n'.format('Filter', name))
            outfd.write('{0:<10} : {1}\n'.format('Flags', str(hook.flags)))
            outfd.write('{0:<10} : {1:#x}\n'.format('Procedure', hook.offPfn))
            outfd.write('{0:<10} : {1}\n'.format('ihmod', hook.ihmod))
            outfd.write('{0:<10} : {1}\n\n'.format('Module', module))
        for (winsta, atom_tables) in data:
            for desk in winsta.desktops():
                for (name, hook) in desk.hooks():
                    module = self.translate_hmod(winsta, atom_tables, hook.ihmod)
                    write_block(outfd, winsta, desk, hook, module, '<any>')
                for thrd in desk.threads():
                    info = '{0} ({1} {2})'.format(thrd.pEThread.Cid.UniqueThread, thrd.ppi.Process.ImageFileName, thrd.ppi.Process.UniqueProcessId)
                    for (name, hook) in thrd.hooks():
                        module = self.translate_hmod(winsta, atom_tables, hook.ihmod)
                        write_block(outfd, winsta, desk, hook, module, info)