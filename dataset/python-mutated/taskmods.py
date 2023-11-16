import os, re
import volatility.plugins.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address, Hex
import volatility.win32 as win32
import volatility.obj as obj
import volatility.debug as debug
import volatility.utils as utils
import volatility.cache as cache

class DllList(common.AbstractWindowsCommand, cache.Testable):
    """Print list of loaded dlls for each process"""

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        cache.Testable.__init__(self)
        config.add_option('OFFSET', short_option='o', default=None, help='EPROCESS offset (in hex) in the physical address space', action='store', type='int')
        config.add_option('PID', short_option='p', default=None, help='Operate on these Process IDs (comma-separated)', action='store', type='str')
        config.add_option('NAME', short_option='n', default=None, help='Operate on these process names (regex)', action='store', type='str')

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Pid', int), ('Base', Address), ('Size', Hex), ('LoadCount', Hex), ('LoadTime', str), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for task in data:
            pid = task.UniqueProcessId
            if task.Peb:
                for m in task.get_load_modules():
                    yield (0, [int(pid), Address(m.DllBase), Hex(m.SizeOfImage), Hex(m.LoadCount), str(m.load_time()), str(m.FullDllName or '')])
            else:
                yield (0, [int(pid), Address(0), Hex(0), Hex(0), '', 'Error reading PEB for pid'])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for task in data:
            pid = task.UniqueProcessId
            outfd.write('*' * 72 + '\n')
            outfd.write('{0} pid: {1:6}\n'.format(task.ImageFileName, pid))
            if task.Peb:
                outfd.write('Command line : {0}\n'.format(str(task.Peb.ProcessParameters.CommandLine or '')))
                outfd.write('{0}\n'.format(str(task.Peb.CSDVersion or '')))
                outfd.write('\n')
                self.table_header(outfd, [('Base', '[addrpad]'), ('Size', '[addr]'), ('LoadCount', '[addr]'), ('LoadTime', '<30'), ('Path', '')])
                for m in task.get_load_modules():
                    self.table_row(outfd, m.DllBase, m.SizeOfImage, m.LoadCount, str(m.load_time()), str(m.FullDllName or ''))
            else:
                outfd.write('Unable to read PEB for task.\n')

    def filter_tasks(self, tasks):
        if False:
            for i in range(10):
                print('nop')
        ' Reduce the tasks based on the user selectable PIDS parameter.\n\n        Returns a reduced list or the full list if config.PIDS not specified.\n        '
        if self._config.PID is not None:
            try:
                pidlist = [int(p) for p in self._config.PID.split(',')]
            except ValueError:
                debug.error('Invalid PID {0}'.format(self._config.PID))
            pids = [t for t in tasks if t.UniqueProcessId in pidlist]
            if len(pids) == 0:
                debug.error('Cannot find PID {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.PID))
            return pids
        if self._config.NAME is not None:
            try:
                name_re = re.compile(self._config.NAME, re.I)
            except re.error:
                debug.error('Invalid name {0}'.format(self._config.NAME))
            names = [t for t in tasks if name_re.search(str(t.ImageFileName))]
            if len(names) == 0:
                debug.error('Cannot find name {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.NAME))
            return names
        return tasks

    @staticmethod
    def virtual_process_from_physical_offset(addr_space, offset):
        if False:
            for i in range(10):
                print('nop')
        ' Returns a virtual process from a physical offset in memory '
        flat_addr_space = utils.load_as(addr_space.get_config(), astype='physical')
        flateproc = obj.Object('_EPROCESS', offset, flat_addr_space)
        tleoffset = addr_space.profile.get_obj_offset('_ETHREAD', 'ThreadListEntry')
        offsets = [tleoffset]
        meta = addr_space.profile.metadata
        major = meta.get('major', 0)
        minor = meta.get('minor', 0)
        build = meta.get('build', 0)
        version = (major, minor, build)
        if meta.get('memory_model') == '64bit' and version == (6, 1, 7601):
            offsets.append(tleoffset + 8)
        for ofs in offsets:
            ethread = obj.Object('_ETHREAD', offset=flateproc.ThreadListHead.Flink.v() - ofs, vm=addr_space)
            virtual_process = ethread.owning_process()
            if virtual_process and offset == addr_space.vtop(virtual_process.obj_offset):
                return virtual_process
        return obj.NoneObject('Unable to bounce back from virtual _ETHREAD to virtual _EPROCESS')

    @cache.CacheDecorator(lambda self: 'tests/pslist/pid={0}/offset={1}'.format(self._config.PID, self._config.OFFSET))
    def calculate(self):
        if False:
            i = 10
            return i + 15
        'Produces a list of processes, or just a single process based on an OFFSET'
        addr_space = utils.load_as(self._config)
        if self._config.OFFSET != None:
            tasks = [self.virtual_process_from_physical_offset(addr_space, self._config.OFFSET)]
        else:
            tasks = self.filter_tasks(win32.tasks.pslist(addr_space))
        return tasks

class PSList(DllList):
    """ Print all running processes by following the EPROCESS lists """

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        DllList.__init__(self, config, *args, **kwargs)
        config.add_option('PHYSICAL-OFFSET', short_option='P', default=False, cache_invalidator=False, help='Display physical offsets instead of virtual', action='store_true')

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        self.table_header(outfd, [('Offset{0}'.format(offsettype), '[addrpad]'), ('Name', '20s'), ('PID', '>6'), ('PPID', '>6'), ('Thds', '>6'), ('Hnds', '>8'), ('Sess', '>6'), ('Wow64', '>6'), ('Start', '30'), ('Exit', '30')])
        for task in data:
            if not self._config.PHYSICAL_OFFSET:
                offset = task.obj_offset
            else:
                offset = task.obj_vm.vtop(task.obj_offset)
            self.table_row(outfd, offset, task.ImageFileName, task.UniqueProcessId, task.InheritedFromUniqueProcessId, task.ActiveThreads, task.ObjectTable.HandleCount, task.SessionId, task.IsWow64, str(task.CreateTime or ''), str(task.ExitTime or ''))

    def render_dot(self, outfd, data):
        if False:
            i = 10
            return i + 15
        objects = set()
        links = set()
        for eprocess in data:
            label = '{0} | {1} |'.format(eprocess.UniqueProcessId, eprocess.ImageFileName)
            if eprocess.ExitTime:
                label += 'exited\\n{0}'.format(eprocess.ExitTime)
                options = ' style = "filled" fillcolor = "lightgray" '
            else:
                label += 'running'
                options = ''
            objects.add('pid{0} [label="{1}" shape="record" {2}];\n'.format(eprocess.UniqueProcessId, label, options))
            links.add('pid{0} -> pid{1} [];\n'.format(eprocess.InheritedFromUniqueProcessId, eprocess.UniqueProcessId))
        outfd.write('digraph processtree { \ngraph [rankdir = "TB"];\n')
        for link in links:
            outfd.write(link)
        for item in objects:
            outfd.write(item)
        outfd.write('}')

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        return TreeGrid([('Offset{0}'.format(offsettype), Address), ('Name', str), ('PID', int), ('PPID', int), ('Thds', int), ('Hnds', int), ('Sess', int), ('Wow64', int), ('Start', str), ('Exit', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for task in data:
            if not self._config.PHYSICAL_OFFSET:
                offset = task.obj_offset
            else:
                offset = task.obj_vm.vtop(task.obj_offset)
            yield (0, [Address(offset), str(task.ImageFileName), int(task.UniqueProcessId), int(task.InheritedFromUniqueProcessId), int(task.ActiveThreads), int(task.ObjectTable.HandleCount), int(task.SessionId), int(task.IsWow64), str(task.CreateTime or ''), str(task.ExitTime or '')])

class MemMap(DllList):
    """Print the memory map"""

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Process', str), ('PID', int), ('Virtual', Address), ('Physical', Address), ('Size', Address), ('DumpFileOffset', Address)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for (pid, task, pagedata) in data:
            task_space = task.get_process_address_space()
            proc = '{0}'.format(task.ImageFileName)
            offset = 0
            if pagedata:
                for p in pagedata:
                    pa = task_space.vtop(p[0])
                    if pa != None:
                        data = task_space.read(p[0], p[1])
                        if data != None:
                            yield (0, [proc, int(pid), Address(p[0]), Address(pa), Address(p[1]), Address(offset)])
                            offset += p[1]

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        first = True
        for (pid, task, pagedata) in data:
            if not first:
                outfd.write('*' * 72 + '\n')
            task_space = task.get_process_address_space()
            outfd.write('{0} pid: {1:6}\n'.format(task.ImageFileName, pid))
            first = False
            offset = 0
            if pagedata:
                self.table_header(outfd, [('Virtual', '[addrpad]'), ('Physical', '[addrpad]'), ('Size', '[addr]'), ('DumpFileOffset', '[addr]')])
                for p in pagedata:
                    pa = task_space.vtop(p[0])
                    if pa != None:
                        data = task_space.read(p[0], p[1])
                        if data != None:
                            self.table_row(outfd, p[0], pa, p[1], offset)
                            offset += p[1]
            else:
                outfd.write('Unable to read pages for task.\n')

    @cache.CacheDecorator(lambda self: 'tests/memmap/pid={0}/offset={1}'.format(self._config.PID, self._config.OFFSET))
    def calculate(self):
        if False:
            i = 10
            return i + 15
        tasks = DllList.calculate(self)
        for task in tasks:
            if task.UniqueProcessId:
                pid = task.UniqueProcessId
                task_space = task.get_process_address_space()
                pages = task_space.get_available_pages()
                yield (pid, task, pages)

class MemDump(MemMap):
    """Dump the addressable memory for a process"""

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        MemMap.__init__(self, config, *args, **kwargs)
        config.add_option('DUMP-DIR', short_option='D', default=None, cache_invalidator=False, help='Directory in which to dump memory')

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        if self._config.DUMP_DIR == None:
            debug.error('Please specify a dump directory (--dump-dir)')
        if not os.path.isdir(self._config.DUMP_DIR):
            debug.error(self._config.DUMP_DIR + ' is not a directory')
        for (pid, task, pagedata) in data:
            outfd.write('*' * 72 + '\n')
            task_space = task.get_process_address_space()
            outfd.write('Writing {0} [{1:6}] to {2}.dmp\n'.format(task.ImageFileName, pid, str(pid)))
            f = open(os.path.join(self._config.DUMP_DIR, str(pid) + '.dmp'), 'wb')
            if pagedata:
                for p in pagedata:
                    data = task_space.read(p[0], p[1])
                    if data == None:
                        if self._config.verbose:
                            outfd.write('Memory Not Accessible: Virtual Address: 0x{0:x} Size: 0x{1:x}\n'.format(p[0], p[1]))
                    else:
                        f.write(data)
            else:
                outfd.write('Unable to read pages for task.\n')
            f.close()