import volatility.plugins.malware.malfind as malfind
import volatility.plugins.mac.pstasks as pstasks
import volatility.plugins.mac.common as common
import volatility.utils as utils
import volatility.debug as debug
import volatility.obj as obj
import re
try:
    import yara
    has_yara = True
except ImportError:
    has_yara = False

class MapYaraScanner(malfind.BaseYaraScanner):
    """A scanner over all memory regions of a process."""

    def __init__(self, task=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Scan the process address space through the VMAs.\n\n        Args:\n          task: The task_struct object for this task.\n        '
        self.task = task
        malfind.BaseYaraScanner.__init__(self, address_space=task.get_process_address_space(), **kwargs)

    def scan(self, offset=0, maxlen=None, max_size=None):
        if False:
            return 10
        for map in self.task.get_proc_maps():
            length = map.links.end - map.links.start
            if max_size and length > max_size:
                debug.warning('Skipping max size entry {0:#x} - {1:#x}'.format(map.links.start, map.links.end))
                continue
            for match in malfind.BaseYaraScanner.scan(self, map.links.start, length):
                yield match

class mac_yarascan(malfind.YaraScan):
    """Scan memory for yara signatures"""

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        malfind.YaraScan.__init__(self, config, *args, **kwargs)
        self._config.add_option('MAX-SIZE', short_option='M', default=1073741824, action='store', type='long', help='Set the maximum size (default is 1GB)')

    @staticmethod
    def is_valid_profile(profile):
        if False:
            print('Hello World!')
        return profile.metadata.get('os', 'Unknown').lower() == 'mac'

    def filter_tasks(self):
        if False:
            while True:
                i = 10
        tasks = pstasks.mac_tasks(self._config).allprocs()
        if self._config.PID is not None:
            try:
                pidlist = [int(p) for p in self._config.PID.split(',')]
            except ValueError:
                debug.error('Invalid PID {0}'.format(self._config.PID))
            pids = [t for t in tasks if t.p_pid in pidlist]
            if len(pids) == 0:
                debug.error('Cannot find PID {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.PID))
            return pids
        if self._config.NAME is not None:
            try:
                name_re = re.compile(self._config.NAME, re.I)
            except re.error:
                debug.error('Invalid name {0}'.format(self._config.NAME))
            names = [t for t in tasks if name_re.search(str(t.p_comm))]
            if len(names) == 0:
                debug.error('Cannot find name {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.NAME))
            return names
        return tasks

    def calculate(self):
        if False:
            i = 10
            return i + 15
        if not has_yara:
            debug.error('Please install Yara from https://plusvic.github.io/yara/')
        rules = self._compile_rules()
        common.set_plugin_members(self)
        if self._config.KERNEL:
            if self.addr_space.profile.metadata.get('memory_model', '32bit') == '32bit':
                if not common.is_64bit_capable(self.addr_space):
                    kernel_start = 0
                else:
                    kernel_start = 3221225472
            else:
                vm_addr = self.addr_space.profile.get_symbol('_vm_min_kernel_address')
                kernel_start = obj.Object('unsigned long', offset=vm_addr, vm=self.addr_space)
            scanner = malfind.DiscontigYaraScanner(rules=rules, address_space=self.addr_space)
            for (hit, address) in scanner.scan(start_offset=kernel_start):
                yield (None, address - self._config.REVERSE, hit, scanner.address_space.zread(address - self._config.REVERSE, self._config.SIZE))
        else:
            tasks = self.filter_tasks()
            for task in tasks:
                if task.p_pid == 0:
                    continue
                scanner = MapYaraScanner(task=task, rules=rules)
                for (hit, address) in scanner.scan(max_size=self._config.MAX_SIZE):
                    yield (task, address - self._config.REVERSE, hit, scanner.address_space.zread(address - self._config.REVERSE, self._config.SIZE))

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for (task, address, hit, buf) in data:
            if task:
                outfd.write('Task: {0} pid {1} rule {2} addr {3:#x}\n'.format(task.p_comm, task.p_pid, hit.rule, address))
            else:
                outfd.write('[kernel] rule {0} addr {1:#x}\n'.format(hit.rule, address))
            outfd.write(''.join(['{0:#018x}  {1:<48}  {2}\n'.format(address + o, h, ''.join(c)) for (o, h, c) in utils.Hexdump(buf)]))