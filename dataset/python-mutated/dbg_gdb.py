from __future__ import print_function
import logging
import threading
import re
import struct
from six.moves.queue import Queue
from voltron.api import *
from voltron.plugin import *
from voltron.dbg import *
try:
    import gdb
    HAVE_GDB = True
except ImportError:
    HAVE_GDB = False
log = logging.getLogger('debugger')
if HAVE_GDB:

    def post_event(func):
        if False:
            while True:
                i = 10
        '\n        Decorator to wrap a GDB adaptor method in a mechanism to run the method\n        on the main thread at the next possible time.\n        '

        def inner(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self.use_post_event:
                q = Queue()

                class Invocation(object):

                    def __call__(killme):
                        if False:
                            print('Hello World!')
                        try:
                            res = func(self, *args, **kwargs)
                        except Exception as e:
                            res = e
                        q.put(res)
                gdb.post_event(Invocation())
                res = q.get()
                if isinstance(res, Exception):
                    raise res
                return res
            else:
                return func(self, *args, **kwargs)
        return inner

    class GDBAdaptor(DebuggerAdaptor):
        archs = {'i386': 'x86', 'i386:intel': 'x86', 'i386:x64-32': 'x86', 'i386:x64-32:intel': 'x86', 'i8086': 'x86', 'i386:x86-64': 'x86_64', 'i386:x86-64:intel': 'x86_64', 'arm': 'arm', 'armv2': 'arm', 'armv2a': 'arm', 'armv3': 'arm', 'armv3m': 'arm', 'armv4': 'arm', 'armv4t': 'arm', 'armv5': 'arm', 'armv5t': 'arm', 'armv5te': 'arm', 'armv7': 'arm', 'armv7s': 'arm', 'powerpc:common': 'powerpc'}
        sizes = {'x86': 4, 'x86_64': 8, 'arm': 4, 'powerpc': 4}
        max_frame = 64
        max_string = 128
        use_post_event = True
        '\n        The interface with an instance of GDB\n        '

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.listeners = []
            self.host_lock = threading.RLock()
            self.host = gdb
            self.busy = False

        def target_is_busy(self, target_id=0):
            if False:
                return 10
            '\n            Returns True or False indicating if the inferior is busy.\n\n            The busy flag is set by the stop and continue handlers registered\n            in the debugger command class.\n            '
            return self.busy

        @post_event
        def version(self):
            if False:
                return 10
            "\n            Get the debugger's version.\n\n            Returns a string containing the debugger's version\n            (e.g. 'GNU gdb (GDB) 7.8')\n            "
            output = gdb.execute('show version', to_string=True)
            try:
                version = output.split('\n')[0]
            except:
                version = None
            return version

        def _target(self, target_id=0):
            if False:
                print('Hello World!')
            '\n            Return information about the specified target.\n\n            Returns data in the following structure:\n            {\n                "id":       0,         # ID that can be used in other funcs\n                "file":     "/bin/ls", # target\'s binary file\n                "arch":     "x86_64",  # target\'s architecture\n                "state:     "stopped"  # state\n            }\n            '
            target = gdb.selected_inferior()
            d = {}
            d['id'] = 0
            d['num'] = target.num
            d['state'] = self._state()
            lines = list(filter(lambda x: x != '', gdb.execute('info inferiors', to_string=True).split('\n')))
            if len(lines) > 1:
                info = list(filter(lambda x: '*' in x[0], map(lambda x: x.split(), lines[1:])))
                d['file'] = info[0][-1]
            else:
                log.debug('No inferiors in `info inferiors`')
                raise NoSuchTargetException()
            d['arch'] = self.get_arch()
            d['byte_order'] = self.get_byte_order()
            d['addr_size'] = self.get_addr_size()
            return d

        @post_event
        def target(self, target_id=0):
            if False:
                while True:
                    i = 10
            '\n            Return information about the current inferior.\n\n            GDB only supports querying the currently selected inferior, rather\n            than an arbitrary target like LLDB, because the API kinda sucks.\n\n            `target_id` is ignored.\n            '
            return self._target()

        @post_event
        def targets(self, target_ids=None):
            if False:
                return 10
            "\n            Return information about the debugger's current targets.\n\n            `target_ids` is ignored. Only the current target is returned. This\n            method is only implemented to maintain API compatibility with the\n            LLDBAdaptor.\n            "
            return [self._target()]

        @validate_target
        @post_event
        def state(self, target_id=0):
            if False:
                print('Hello World!')
            '\n            Get the state of a given target.\n            '
            return self._state()

        @validate_busy
        @validate_target
        @post_event
        def registers(self, target_id=0, thread_id=None, registers=[]):
            if False:
                return 10
            '\n            Get the register values for a given target/thread.\n            '
            arch = self.get_arch()
            log.debug('xxx')
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
                regs = {}
                for reg in registers:
                    regs[reg] = self.get_register(reg)
            else:
                log.debug('Getting registers for arch {}'.format(arch))
                if arch == 'x86_64':
                    regs = self.get_registers_x86_64()
                elif arch == 'x86':
                    regs = self.get_registers_x86()
                elif arch == 'arm':
                    regs = self.get_registers_arm()
                elif arch == 'powerpc':
                    regs = self.get_registers_powerpc()
                else:
                    raise UnknownArchitectureException()
            return regs

        @validate_busy
        @validate_target
        @post_event
        def stack_pointer(self, target_id=0, thread_id=None):
            if False:
                while True:
                    i = 10
            '\n            Get the value of the stack pointer register.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                sp_name = self.reg_names[arch]['sp']
                sp = self.get_register(sp_name)
            else:
                raise UnknownArchitectureException()
            return (sp_name, sp)

        @validate_busy
        @validate_target
        @post_event
        def program_counter(self, target_id=0, thread_id=None):
            if False:
                i = 10
                return i + 15
            '\n            Get the value of the program counter register.\n            '
            return self._program_counter(target_id, thread_id)

        def _program_counter(self, target_id=0, thread_id=None):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Implementation of getting PC to avoid recursive decorators\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                pc_name = self.reg_names[arch]['pc']
                pc = self.get_register(pc_name)
            else:
                raise UnknownArchitectureException()
            return (pc_name, pc)

        @validate_busy
        @validate_target
        @post_event
        def memory(self, address, length, target_id=0):
            if False:
                print('Hello World!')
            '\n            Read memory from the inferior.\n\n            `address` is the address at which to start reading\n            `length` is the number of bytes to read\n            '
            log.debug('Reading 0x{:x} bytes of memory at 0x{:x}'.format(length, address))
            memory = bytes(gdb.selected_inferior().read_memory(address, length))
            return memory

        @validate_busy
        @validate_target
        @post_event
        def write_memory(self, address, data, target_id=0):
            if False:
                print('Hello World!')
            "\n            Write to the inferior's memory.\n\n            `address` is the address at which to start write\n            `data` is the data to write\n            "
            log.debug('Writing 0x{:x} bytes of memory at 0x{:x}'.format(len(data), address))
            memory = bytes(gdb.selected_inferior().write_memory(address, data))

        @validate_busy
        @validate_target
        @post_event
        def disassemble(self, target_id=0, address=None, count=16):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get a disassembly of the instructions at the given address.\n\n            `address` is the address at which to disassemble. If None, the\n            current program counter is used.\n            `count` is the number of instructions to disassemble.\n            '
            if address == None:
                (pc_name, address) = self._program_counter(target_id=target_id)
            output = gdb.execute('x/{}i 0x{:x}'.format(count, address), to_string=True)
            return output

        @validate_busy
        @validate_target
        @post_event
        def dereference(self, pointer, target_id=0):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Recursively dereference a pointer for display\n            '
            if isinstance(pointer, six.integer_types):
                fmt = ('<' if self.get_byte_order() == 'little' else '>') + {2: 'H', 4: 'L', 8: 'Q'}[self.get_addr_size()]
                addr = pointer
                chain = []
                while True:
                    try:
                        mem = gdb.selected_inferior().read_memory(addr, self.get_addr_size())
                        (ptr,) = struct.unpack(fmt, mem)
                        if ptr in [x[1] for x in chain]:
                            break
                        chain.append(('pointer', addr))
                        addr = ptr
                    except gdb.MemoryError:
                        log.exception('Dereferencing pointer 0x{:X}'.format(addr))
                        break
                    except OverflowError:
                        log.exception('Dereferencing pointer 0x{:X}'.format(addr))
                        break
                if len(chain):
                    (p, addr) = chain[-1]
                    output = gdb.execute('info symbol 0x{:x}'.format(addr), to_string=True)
                    log.debug('output = {}'.format(output))
                    if 'No symbol matches' not in output:
                        chain.append(('symbol', output.strip()))
                        log.debug('symbol context: {}'.format(str(chain[-1])))
                    else:
                        log.debug('no symbol context, trying as a string')
                        mem = gdb.selected_inferior().read_memory(addr, 2)
                        if ord(mem[0]) <= 127 and ord(mem[0]) != 0:
                            a = []
                            for i in range(0, self.max_string):
                                mem = gdb.selected_inferior().read_memory(addr + i, 1)
                                if ord(mem[0]) == 0 or ord(mem[0]) > 127:
                                    break
                                if isinstance(mem, memoryview):
                                    a.append(mem.tobytes().decode('latin1'))
                                else:
                                    a.append(str(mem))
                            chain.append(('string', ''.join(a)))
                log.debug('chain: {}'.format(chain))
            else:
                chain = []
            return chain

        @post_event
        def command(self, command=None):
            if False:
                print('Hello World!')
            '\n            Execute a command in the debugger.\n\n            `command` is the command string to execute.\n            '
            if command:
                res = gdb.execute(command, to_string=True)
            else:
                raise Exception('No command specified')
            return res

        @post_event
        def disassembly_flavor(self):
            if False:
                return 10
            "\n            Return the disassembly flavor setting for the debugger.\n\n            Returns 'intel' or 'att'\n            "
            flavor = re.search('flavor is "(.*)"', gdb.execute('show disassembly-flavor', to_string=True)).group(1)
            return flavor

        @post_event
        def breakpoints(self, target_id=0):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return a list of breakpoints.\n\n            Returns data in the following structure:\n            [\n                {\n                    "id":           1,\n                    "enabled":      True,\n                    "one_shot":     False,\n                    "hit_count":    5,\n                    "locations": [\n                        {\n                            "address":  0x100000cf0,\n                            "name":     \'main\'\n                        }\n                    ]\n                }\n            ]\n            '
            breakpoints = []
            for b in gdb.breakpoints() or ():
                try:
                    if b.location.startswith('*'):
                        addr = int(b.location[1:], 16)
                    else:
                        output = gdb.execute('info addr {}'.format(b.location), to_string=True)
                        m = re.match('.*is at ([^ ]*) .*', output)
                        if not m:
                            m = re.match('.*at address ([^ ]*)\\..*', output)
                        if m:
                            addr = int(m.group(1), 16)
                        else:
                            addr = 0
                except:
                    addr = 0
                breakpoints.append({'id': b.number, 'enabled': b.enabled, 'one_shot': b.temporary, 'hit_count': b.hit_count, 'locations': [{'address': addr, 'name': b.location}]})
            return breakpoints

        @post_event
        def backtrace(self, target_id=0, thread_id=None):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return a list of stack frames.\n            '
            frames = []
            f = gdb.newest_frame()
            for i in range(self.max_frame):
                if not f:
                    break
                frames.append({'index': i, 'addr': f.pc(), 'name': f.name()})
                f = f.older()
            return frames

        def capabilities(self):
            if False:
                print('Hello World!')
            "\n            Return a list of the debugger's capabilities.\n\n            Thus far only the 'async' capability is supported. This indicates\n            that the debugger host can be queried from a background thread,\n            and that views can use non-blocking API requests without queueing\n            requests to be dispatched next time the debugger stops.\n            "
            return ['async']

        def _state(self):
            if False:
                print('Hello World!')
            '\n            Get the state of a given target. Internal use.\n            '
            target = gdb.selected_inferior()
            if target.is_valid():
                try:
                    output = gdb.execute('info program', to_string=True)
                    if 'not being run' in output:
                        state = 'invalid'
                    elif 'stopped' in output:
                        state = 'stopped'
                except gdb.error as e:
                    if 'Selected thread is running.' == str(e):
                        state = 'running'
            else:
                state = 'invalid'
            return state

        def get_register(self, reg_name):
            if False:
                while True:
                    i = 10
            arch = self.get_arch()
            if arch == 'x86_64':
                reg = self.get_register_x86_64(reg_name)
            elif arch == 'x86':
                reg = self.get_register_x86(reg_name)
            elif arch == 'arm':
                reg = self.get_register_arm(reg_name)
            elif arch == 'powerpc':
                reg = self.get_register_powerpc(reg_name)
            else:
                raise UnknownArchitectureException()
            return reg

        def get_registers_x86_64(self):
            if False:
                return 10
            regs = ['rax', 'rbx', 'rcx', 'rdx', 'rbp', 'rsp', 'rdi', 'rsi', 'rip', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'cs', 'ds', 'es', 'fs', 'gs', 'ss']
            vals = {}
            for reg in regs:
                try:
                    vals[reg] = self.get_register_x86_64(reg)
                except:
                    log.debug('Failed getting reg: ' + reg)
                    vals[reg] = 'N/A'
            try:
                vals['rflags'] = int(gdb.execute('info reg $eflags', to_string=True).split()[1], 16)
            except:
                log.debug('Failed getting reg: eflags')
                vals['rflags'] = 'N/A'
            try:
                sse = self.get_registers_sse(16)
                vals = dict(list(vals.items()) + list(sse.items()))
            except gdb.error:
                log.exception('Failed to get SSE registers')
            try:
                fpu = self.get_registers_fpu()
                vals = dict(list(vals.items()) + list(fpu.items()))
            except gdb.error:
                log.exception('Failed to get FPU registers')
            return vals

        def get_register_x86_64(self, reg):
            if False:
                i = 10
                return i + 15
            return int(gdb.parse_and_eval('(long long)$' + reg)) & 18446744073709551615

        def get_registers_x86(self):
            if False:
                i = 10
                return i + 15
            regs = ['eax', 'ebx', 'ecx', 'edx', 'ebp', 'esp', 'edi', 'esi', 'eip', 'cs', 'ds', 'es', 'fs', 'gs', 'ss']
            vals = {}
            for reg in regs:
                try:
                    vals[reg] = self.get_register_x86(reg)
                except:
                    log.debug('Failed getting reg: ' + reg)
                    vals[reg] = 'N/A'
            try:
                vals['eflags'] = int(gdb.execute('info reg $eflags', to_string=True).split()[1], 16)
            except:
                log.debug('Failed getting reg: eflags')
                vals['eflags'] = 'N/A'
            try:
                sse = self.get_registers_sse(8)
                vals = dict(list(vals.items()) + list(sse.items()))
            except gdb.error:
                log.exception('Failed to get SSE registers')
            try:
                fpu = self.get_registers_fpu()
                vals = dict(list(vals.items()) + list(fpu.items()))
            except gdb.error:
                log.exception('Failed to get SSE registers')
            return vals

        def get_register_x86(self, reg):
            if False:
                print('Hello World!')
            log.debug('Getting register: ' + reg)
            return int(gdb.parse_and_eval('(long)$' + reg)) & 4294967295

        def get_registers_sse(self, num=8):
            if False:
                print('Hello World!')
            regs = {}
            for line in gdb.execute('info all-registers', to_string=True).split('\n'):
                m = re.match('^([xyz]mm\\d+)\\s.*uint128 = (0x[0-9a-f]+)\\}', line)
                if m:
                    regs[m.group(1)] = int(m.group(2), 16)
            return regs

        def get_registers_fpu(self):
            if False:
                for i in range(10):
                    print('nop')
            regs = {}
            for i in range(8):
                reg = 'st' + str(i)
                try:
                    regs[reg] = int(gdb.execute('info reg ' + reg, to_string=True).split()[-1][2:-1], 16)
                except:
                    log.debug('Failed getting reg: ' + reg)
                    regs[reg] = 'N/A'
            return regs

        def get_registers_arm(self):
            if False:
                i = 10
                return i + 15
            log.debug('Getting registers')
            regs = ['pc', 'sp', 'lr', 'cpsr', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12']
            vals = {}
            for reg in regs:
                try:
                    vals[reg] = self.get_register_arm(reg)
                except:
                    log.debug('Failed getting reg: ' + reg)
                    vals[reg] = 'N/A'
            return vals

        def get_register_arm(self, reg):
            if False:
                i = 10
                return i + 15
            log.debug('Getting register: ' + reg)
            return int(gdb.parse_and_eval('(long)$' + reg)) & 4294967295

        def get_registers_powerpc(self):
            if False:
                print('Hello World!')
            log.debug('Getting registers')
            regs = ['pc', 'msr', 'cr', 'lr', 'ctr', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30', 'r31']
            vals = {}
            for reg in regs:
                try:
                    vals[reg] = self.get_register_powerpc(reg)
                except:
                    log.debug('Failed getting reg: ' + reg)
                    vals[reg] = 'N/A'
            return vals

        def get_register_powerpc(self, reg):
            if False:
                for i in range(10):
                    print('nop')
            log.debug('Getting register: ' + reg)
            return int(gdb.parse_and_eval('(long)$' + reg)) & 4294967295

        def get_next_instruction(self):
            if False:
                i = 10
                return i + 15
            return self.get_disasm().split('\n')[0].split(':')[1].strip()

        def get_arch(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                arch = gdb.selected_frame().architecture().name()
            except:
                arch = re.search('\\(currently (.*)\\)', gdb.execute('show architecture', to_string=True)).group(1)
            return self.archs[arch]

        def get_addr_size(self):
            if False:
                for i in range(10):
                    print('nop')
            arch = self.get_arch()
            return self.sizes[arch]

        def get_byte_order(self):
            if False:
                print('Hello World!')
            return 'little' if 'little' in gdb.execute('show endian', to_string=True) else 'big'

    class GDBCommand(DebuggerCommand, gdb.Command):
        """
        Debugger command class for GDB
        """

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super(GDBCommand, self).__init__('voltron', gdb.COMMAND_NONE, gdb.COMPLETE_NONE)
            self.adaptor = voltron.debugger
            self.registered = False
            self.register_hooks()

        def invoke(self, arg, from_tty):
            if False:
                return 10
            self.handle_command(arg)

        def register_hooks(self):
            if False:
                for i in range(10):
                    print('nop')
            if not self.registered:
                gdb.events.stop.connect(self.stop_handler)
                gdb.events.exited.connect(self.stop_and_exit_handler)
                gdb.events.cont.connect(self.cont_handler)
                self.registered = True

        def unregister_hooks(self):
            if False:
                i = 10
                return i + 15
            if self.registered:
                gdb.events.stop.disconnect(self.stop_handler)
                gdb.events.exited.disconnect(self.stop_and_exit_handler)
                gdb.events.cont.disconnect(self.cont_handler)
                self.registered = False

        def stop_handler(self, event):
            if False:
                for i in range(10):
                    print('nop')
            self.adaptor.update_state()
            voltron.debugger.busy = False
            voltron.server.dispatch_queue()
            log.debug('Inferior stopped')

        def exit_handler(self, event):
            if False:
                return 10
            log.debug('Inferior exited')
            voltron.debugger.busy = False

        def stop_and_exit_handler(self, event):
            if False:
                return 10
            log.debug('Inferior stopped and exited')
            voltron.debugger.busy = False
            self.stop_handler(event)
            self.exit_handler(event)

        def cont_handler(self, event):
            if False:
                i = 10
                return i + 15
            log.debug('Inferior continued')
            voltron.debugger.busy = True

    class GDBAdaptorPlugin(DebuggerAdaptorPlugin):
        host = 'gdb'
        adaptor_class = GDBAdaptor
        command_class = GDBCommand