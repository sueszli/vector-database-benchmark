from __future__ import print_function
import logging
import threading
import array
from voltron.api import *
from voltron.plugin import *
from voltron.dbg import *
try:
    in_windbg = False
    import pykd
    try:
        import vtrace
    except:
        in_windbg = True
except ImportError:
    pass
log = logging.getLogger('debugger')
if in_windbg:

    class WinDbgAdaptor(DebuggerAdaptor):
        sizes = {'x86': 4, 'x86_64': 8}
        max_deref = 24
        max_string = 128

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self.listeners = []
            self.host_lock = threading.RLock()
            self.host = pykd

        def version(self):
            if False:
                print('Hello World!')
            "\n            Get the debugger's version.\n\n            Returns a string containing the debugger's version\n            (e.g. 'Microsoft (R) Windows Debugger Version whatever, pykd 0.3.0.38')\n            "
            try:
                [windbg] = [line for line in pykd.dbgCommand('version').split('\n') if 'Microsoft (R) Windows Debugger Version' in line]
            except:
                windbg = 'WinDbg <unknown>'
            return '{}, {}'.format(windbg, 'pykd {}'.format(pykd.version))

        def _target(self, target_id=0):
            if False:
                i = 10
                return i + 15
            '\n            Return information about the specified target.\n\n            Returns data in the following structure:\n            {\n                "id":       0,         # ID that can be used in other funcs\n                "file":     "/bin/ls", # target\'s binary file\n                "arch":     "x86_64",  # target\'s architecture\n                "state:     "stopped"  # state\n            }\n            '
            d = {}
            d['id'] = pykd.getCurrentProcessId()
            d['num'] = d['id']
            d['state'] = self._state()
            d['file'] = pykd.getProcessExeName()
            d['arch'] = self.get_arch()
            d['byte_order'] = self.get_byte_order()
            d['addr_size'] = self.get_addr_size()
            return d

        @lock_host
        def target(self, target_id=0):
            if False:
                print('Hello World!')
            '\n            Return information about the current inferior.\n\n            We only support querying the current inferior with WinDbg.\n\n            `target_id` is ignored.\n            '
            return self._target()

        @lock_host
        def targets(self, target_ids=None):
            if False:
                while True:
                    i = 10
            "\n            Return information about the debugger's current targets.\n\n            `target_ids` is ignored. Only the current target is returned. This\n            method is only implemented to maintain API compatibility with the\n            LLDBAdaptor.\n            "
            return [self._target()]

        @validate_target
        @lock_host
        def state(self, target_id=0):
            if False:
                i = 10
                return i + 15
            '\n            Get the state of a given target.\n            '
            return self._state()

        @validate_busy
        @validate_target
        @lock_host
        def registers(self, target_id=0, thread_id=None, registers=[]):
            if False:
                i = 10
                return i + 15
            '\n            Get the register values for a given target/thread.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                if 'pc' in registers:
                    registers.remove('pc')
                    registers.append(self.reg_names[arch]['pc'])
                if 'sp' in registers:
                    registers.remove('sp')
                    registers.append(self.reg_names[arch]['sp'])
            else:
                raise Exception('Unsupported architecture: {}'.format(target['arch']))
            if registers != []:
                vals = {}
                for reg in registers:
                    vals[reg] = pykd.reg(reg)
            else:
                log.debug('Getting registers for arch {}'.format(arch))
                if arch == 'x86_64':
                    reg_names = ['rax', 'rbx', 'rcx', 'rdx', 'rbp', 'rsp', 'rdi', 'rsi', 'rip', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'cs', 'ds', 'es', 'fs', 'gs', 'ss']
                elif arch == 'x86':
                    reg_names = ['eax', 'ebx', 'ecx', 'edx', 'ebp', 'esp', 'edi', 'esi', 'eip', 'cs', 'ds', 'es', 'fs', 'gs', 'ss']
                else:
                    raise UnknownArchitectureException()
                vals = {}
                for reg in reg_names:
                    try:
                        vals[reg] = pykd.reg(reg)
                    except:
                        log.debug('Failed getting reg: ' + reg)
                        vals[reg] = 'N/A'
                try:
                    vals['rflags'] = pykd.reg(reg)
                except:
                    log.debug('Failed getting reg: eflags')
                    vals['rflags'] = 'N/A'
                try:
                    vals.update(self.get_registers_sse(16))
                except:
                    log.exception('Failed to get SSE registers')
                try:
                    vals.update(self.get_registers_fpu())
                except:
                    log.exception('Failed to get FPU registers')
            return vals

        @validate_busy
        @validate_target
        @lock_host
        def stack_pointer(self, target_id=0, thread_id=None):
            if False:
                print('Hello World!')
            '\n            Get the value of the stack pointer register.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                sp_name = self.reg_names[arch]['sp']
                sp = pykd.reg(sp_name)
            else:
                raise UnknownArchitectureException()
            return (sp_name, sp)

        @validate_busy
        @validate_target
        @lock_host
        def program_counter(self, target_id=0, thread_id=None):
            if False:
                while True:
                    i = 10
            '\n            Get the value of the program counter register.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                pc_name = self.reg_names[arch]['pc']
                pc = pykd.reg(pc_name)
            else:
                raise UnknownArchitectureException()
            return (pc_name, pc)

        @validate_busy
        @validate_target
        @lock_host
        def memory(self, address, length, target_id=0):
            if False:
                print('Hello World!')
            '\n            Get the register values for .\n\n            `address` is the address at which to start reading\n            `length` is the number of bytes to read\n            '
            log.debug('Reading 0x{:x} bytes of memory at 0x{:x}'.format(length, address))
            memory = array.array('B', pykd.loadBytes(address, length)).tostring()
            return memory

        @validate_busy
        @validate_target
        @lock_host
        def stack(self, length, target_id=0, thread_id=None):
            if False:
                print('Hello World!')
            '\n            Get the register values for .\n\n            `length` is the number of bytes to read\n            `target_id` is a target ID (or None for the first target)\n            `thread_id` is a thread ID (or None for the selected thread)\n            '
            (sp_name, sp) = self.stack_pointer(target_id=target_id, thread_id=thread_id)
            memory = self.memory(sp, length, target_id=target_id)
            return memory

        @validate_busy
        @validate_target
        @lock_host
        def disassemble(self, target_id=0, address=None, count=16):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get a disassembly of the instructions at the given address.\n\n            `address` is the address at which to disassemble. If None, the\n            current program counter is used.\n            `count` is the number of instructions to disassemble.\n            '
            if address is None:
                (pc_name, address) = self.program_counter(target_id=target_id)
            output = pykd.dbgCommand('u 0x{:x} l{}'.format(address, count))
            return output

        @validate_busy
        @validate_target
        @lock_host
        def dereference(self, pointer, target_id=0):
            if False:
                i = 10
                return i + 15
            '\n            Recursively dereference a pointer for display\n            '
            fmt = ('<' if self.get_byte_order() == 'little' else '>') + {2: 'H', 4: 'L', 8: 'Q'}[self.get_addr_size()]
            addr = pointer
            chain = []
            for i in range(0, self.max_deref):
                try:
                    [ptr] = pykd.loadPtrs(addr, 1)
                    if ptr in chain:
                        break
                    chain.append(('pointer', addr))
                    addr = ptr
                except:
                    log.exception('Dereferencing pointer 0x{:X}'.format(addr))
                    break
            if len(chain):
                (p, addr) = chain[-1]
                output = pykd.findSymbol(addr)
                sym = True
                try:
                    if int(output, 16) == addr:
                        sym = False
                        log.debug('no symbol context')
                except:
                    pass
                if sym:
                    chain.append(('symbol', output.strip()))
                else:
                    log.debug('no symbol context')
                    mem = pykd.loadBytes(addr, 2)
                    if mem[0] < 127:
                        if mem[1] == 0:
                            a = []
                            for i in range(0, self.max_string, 2):
                                mem = pykd.loadBytes(addr + i, 2)
                                if mem == [0, 0]:
                                    break
                                a.extend(mem)
                            output = array.array('B', a).tostring().decode('UTF-16').encode('latin1')
                            chain.append(('unicode', output))
                        else:
                            output = pykd.loadCStr(addr)
                            chain.append(('string', output))
            log.debug('chain: {}'.format(chain))
            return chain

        @lock_host
        def command(self, command=None):
            if False:
                while True:
                    i = 10
            '\n            Execute a command in the debugger.\n\n            `command` is the command string to execute.\n            '
            if command:
                res = pykd.dbgCommand(command)
            else:
                raise Exception('No command specified')
            return res

        @lock_host
        def disassembly_flavor(self):
            if False:
                while True:
                    i = 10
            "\n            Return the disassembly flavor setting for the debugger.\n\n            Returns 'intel' or 'att'\n            "
            return 'intel'

        @lock_host
        def breakpoints(self, target_id=0):
            if False:
                return 10
            '\n            Return a list of breakpoints.\n\n            Returns data in the following structure:\n            [\n                {\n                    "id":           1,\n                    "enabled":      True,\n                    "one_shot":     False,\n                    "hit_count":    5,\n                    "locations": [\n                        {\n                            "address":  0x100000cf0,\n                            "name":     \'main\'\n                        }\n                    ]\n                }\n            ]\n            '
            breakpoints = []
            for i in range(0, pykd.getNumberBreakpoints()):
                b = pykd.getBp(i)
                addr = b.getOffset()
                name = hex(addr)
                try:
                    name = pykd.findSymbol(addr)
                except:
                    log.exception('No symbol found for address {}'.format(addr))
                    pass
                breakpoints.append({'id': i, 'enabled': True, 'one_shot': False, 'hit_count': '-', 'locations': [{'address': addr, 'name': name}]})
            return breakpoints

        def capabilities(self):
            if False:
                print('Hello World!')
            "\n            Return a list of the debugger's capabilities.\n\n            Thus far only the 'async' capability is supported. This indicates\n            that the debugger host can be queried from a background thread,\n            and that views can use non-blocking API requests without queueing\n            requests to be dispatched next time the debugger stops.\n            "
            return ['async']

        def _state(self):
            if False:
                while True:
                    i = 10
            '\n            Get the state of a given target. Internal use.\n            '
            s = pykd.getExecutionStatus()
            if s == pykd.executionStatus.Break:
                state = 'stopped'
            elif s == pykd.executionStatus.Go:
                state = 'running'
            else:
                state = 'invalid'
            return state

        def get_registers_sse(self, num=8):
            if False:
                print('Hello World!')
            regs = {}
            for i in range(0, 16):
                try:
                    reg = 'xmm{}'.format(i)
                    regs[reg] = pykd.reg(reg)
                except:
                    break
            return regs

        def get_registers_fpu(self):
            if False:
                while True:
                    i = 10
            regs = {}
            for i in range(0, 8):
                try:
                    reg = 'st{}'.format(i)
                    regs[reg] = pykd.reg(reg)
                except:
                    break
            return regs

        def get_next_instruction(self):
            if False:
                print('Hello World!')
            return str(pykd.disasm())

        def get_arch(self):
            if False:
                i = 10
                return i + 15
            t = pykd.getCPUType()
            if t == pykd.CPUType.I386:
                return 'x86'
            else:
                return 'x86_64'
            return arch

        def get_addr_size(self):
            if False:
                print('Hello World!')
            arch = self.get_arch()
            return self.sizes[arch]

        def get_byte_order(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'little'

    class EventHandler(pykd.eventHandler):
        """
        Event handler for WinDbg/PyKD events.
        """

        def __init__(self, adaptor, *args, **kwargs):
            if False:
                return 10
            super(EventHandler, self).__init__(*args, **kwargs)
            self.adaptor = adaptor

        def onExecutionStatusChange(self, status):
            if False:
                while True:
                    i = 10
            if status == pykd.executionStatus.Break:
                self.adaptor.update_state()
                voltron.server.dispatch_queue()

    class WinDbgCommand(DebuggerCommand):
        """
        Debugger command class for WinDbg
        """

        def __init__(self):
            if False:
                return 10
            super(WinDbgCommand, self).__init__()
            self.register_hooks()

        def invoke(self, debugger, command, result, dict):
            if False:
                return 10
            self.handle_command(command)

        def register_hooks(self):
            if False:
                while True:
                    i = 10
            self.handler = EventHandler(self.adaptor)

        def unregister_hooks(self):
            if False:
                return 10
            del self.handler
            self.handler = None

    class WinDbgAdaptorPlugin(DebuggerAdaptorPlugin):
        host = 'windbg'
        adaptor_class = WinDbgAdaptor
        command_class = WinDbgCommand