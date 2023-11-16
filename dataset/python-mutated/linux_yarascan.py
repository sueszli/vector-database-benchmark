import volatility.plugins.malware.malfind as malfind
import volatility.plugins.linux.pslist as pslist
import volatility.plugins.linux.common as linux_common
import volatility.utils as utils
import volatility.debug as debug
import re
try:
    import yara
    has_yara = True
except ImportError:
    has_yara = False

class VmaYaraScanner(malfind.BaseYaraScanner):
    """A scanner over all memory regions of a process."""

    def __init__(self, task=None, **kwargs):
        if False:
            return 10
        'Scan the process address space through the VMAs.\n\n        Args:\n          task: The task_struct object for this task.\n        '
        self.task = task
        malfind.BaseYaraScanner.__init__(self, address_space=task.get_process_address_space(), **kwargs)

    def scan(self, offset=0, maxlen=None):
        if False:
            return 10
        for vma in self.task.get_proc_maps():
            for match in malfind.BaseYaraScanner.scan(self, vma.vm_start, vma.vm_end - vma.vm_start):
                yield match

class linux_yarascan(malfind.YaraScan):
    """A shell in the Linux memory image"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            print('Hello World!')
        return profile.metadata.get('os', 'Unknown').lower() == 'linux'

    def filter_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        tasks = pslist.linux_pslist(self._config).calculate()
        if self._config.PID is not None:
            try:
                pidlist = [int(p) for p in self._config.PID.split(',')]
            except ValueError:
                debug.error('Invalid PID {0}'.format(self._config.PID))
            pids = [t for t in tasks if t.pid in pidlist]
            if len(pids) == 0:
                debug.error('Cannot find PID {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.PID))
            return pids
        if self._config.NAME is not None:
            try:
                name_re = re.compile(self._config.NAME, re.I)
            except re.error:
                debug.error('Invalid name {0}'.format(self._config.NAME))
            names = [t for t in tasks if name_re.search(str(t.comm))]
            if len(names) == 0:
                debug.error('Cannot find name {0}. If its terminated or unlinked, use psscan and then supply --offset=OFFSET'.format(self._config.NAME))
            return names
        return tasks

    def calculate(self):
        if False:
            while True:
                i = 10
        if not has_yara:
            debug.error('Please install Yara from https://plusvic.github.io/yara/')
        rules = self._compile_rules()
        linux_common.set_plugin_members(self)
        if self._config.KERNEL:
            if self.addr_space.profile.metadata.get('memory_model', '32bit') == '32bit':
                kernel_start = 3221225472
            else:
                kernel_start = 18446744071562067968
            scanner = malfind.DiscontigYaraScanner(rules=rules, address_space=self.addr_space)
            for (hit, address) in scanner.scan(start_offset=kernel_start):
                yield (None, address - self._config.REVERSE, hit, scanner.address_space.zread(address - self._config.REVERSE, self._config.SIZE))
        else:
            tasks = self.filter_tasks()
            for task in tasks:
                scanner = VmaYaraScanner(task=task, rules=rules)
                for (hit, address) in scanner.scan():
                    yield (task, address - self._config.REVERSE, hit, scanner.address_space.zread(address - self._config.REVERSE, self._config.SIZE))

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for (task, address, hit, buf) in data:
            if task:
                outfd.write('Task: {0} pid {1} rule {2} addr {3:#x}\n'.format(task.comm, task.pid, hit.rule, address))
            else:
                outfd.write('[kernel] rule {0} addr {1:#x}\n'.format(hit.rule, address))
            outfd.write(''.join(['{0:#010x}  {1:<48}  {2}\n'.format(address + o, h, ''.join(c)) for (o, h, c) in utils.Hexdump(buf)]))