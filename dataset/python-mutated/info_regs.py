"""
@author: Mariano `emdel` Graziano, Edwin Smulders
@license: GNU General Public License 2.0 or later
@contact: graziano@eurecom.fr, mail@edwinsmulders.eu
"""
import volatility.plugins.linux.common as common
import volatility.plugins.linux.pslist as linux_pslist
import collections
import struct
import volatility.debug as debug
offsets = {}
offsets['32bit'] = ['ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'eax', 'ds', 'es', 'fs', 'gs', 'orig_eax', 'eip', 'cs', 'eflags', 'esp', 'ss']
offsets['64bit'] = ['r15', 'r14', 'r13', 'r12', 'rbp', 'rbx', 'r11', 'r10', 'r9', 'r8', 'rax', 'rcx', 'rdx', 'rsi', 'rdi', 'unknown', 'rip', 'cs', 'eflags', 'rsp', 'ss']
reg_size = {}
reg_size['32bit'] = 4
reg_size['64bit'] = 8
fmt = {}
fmt['32bit'] = '<I'
fmt['64bit'] = '<Q'

class linux_info_regs(linux_pslist.linux_pslist):
    """It's like 'info registers' in GDB. It prints out all the
processor registers involved during the context switch."""

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        linux_pslist.linux_pslist.__init__(self, config, *args, **kwargs)
        self.bits = 0
        self.reg_size = 0
        self.offsets = []
        self.fmt = ''

    def calculate(self):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        if self.profile.metadata['arch'] not in ['x64', 'x86']:
            debug.error('This plugin is only supported on Intel-based memory captures')
        self.bits = self.profile.metadata.get('memory_model', '32bit')
        self.reg_size = reg_size[self.bits]
        self.offsets = offsets[self.bits]
        self.fmt = fmt[self.bits]
        for proc in linux_pslist.linux_pslist(self._config).calculate():
            name = proc.get_commandline()
            thread_registers = []
            for thread_task in proc.threads():
                thread_name = thread_task.comm
                regs = self.parse_kernel_stack(thread_task)
                thread_registers.append((thread_name, regs))
            yield (proc, name, thread_registers)

    def render_text(self, outfd, data):
        if False:
            return 10
        for (task, name, thread_regs) in data:
            outfd.write('Process Name: {} - PID: {}\n'.format(name, str(task.pid)))
            outfd.write('Registers (per thread):\n')
            fmt = str(2 * self.reg_size)
            for (thread_name, regs) in thread_regs:
                outfd.write('  Thread Name: {}\n'.format(thread_name))
                if regs != None:
                    for (reg, value) in regs.items():
                        outfd.write(('    {:8s}: {:0' + fmt + 'x}\n').format(reg, value))

    def parse_kernel_stack(self, task):
        if False:
            while True:
                i = 10
        result = collections.OrderedDict()
        if 1 or task.mm:
            sp0 = task.thread.sp0
            addr = sp0
            for reg in self.offsets[::-1]:
                addr -= self.reg_size
                val_raw = self.addr_space.read(addr, self.reg_size)
                val = struct.unpack(self.fmt, val_raw)[0]
                result[reg] = val
            return result
        return None