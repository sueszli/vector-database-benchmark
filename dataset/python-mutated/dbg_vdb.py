from __future__ import print_function
import struct
import string
import logging
import threading
from voltron.api import *
from voltron.plugin import *
from voltron.dbg import *
try:
    import vtrace
    import vdb
    import envi
    HAVE_VDB = True
except ImportError:
    HAVE_VDB = False
log = logging.getLogger('debugger')
if HAVE_VDB:

    class NotAStringError(Exception):
        pass

    class FailedToReadMemoryError(Exception):
        pass

    class VDBAdaptor(DebuggerAdaptor):
        """
        The interface with an instance of VDB
        """
        archs = {'i386': 'x86', 'amd64': 'x86_64', 'arm': 'arm'}
        sizes = {'x86': 4, 'x86_64': 8, 'arm': 4}
        reg_names = {'x86_64': {'pc': 'rip', 'sp': 'rsp'}, 'x86': {'pc': 'eip', 'sp': 'esp'}}

        def __init__(self, host, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.listeners = []
            self.host_lock = threading.RLock()
            self._vdb = host
            self._vtrace = vtrace

        def version(self):
            if False:
                for i in range(10):
                    print('nop')
            "\n            Get the debugger's version.\n\n            Returns a string containing the debugger's version\n            (e.g. 'GNU gdb (GDB) 7.8')\n            "
            return 'VDB/version-unknown'

        def _target(self, target_id=0):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return information about the specified target.\n\n            Returns data in the following structure:\n            {\n                "id":       0,         # ID that can be used in other funcs\n                "file":     "/bin/ls", # target\'s binary file\n                "arch":     "x86_64",  # target\'s architecture\n                "state:     "stopped"  # state\n            }\n            '
            d = {}
            d['id'] = 0
            d['state'] = self._state()
            d['file'] = self._vdb.getTrace().metadata['ExeName']
            d['arch'] = self.get_arch()
            d['byte_order'] = self.get_byte_order()
            d['addr_size'] = self.get_addr_size()
            return d

        @lock_host
        def target(self, target_id=0):
            if False:
                print('Hello World!')
            '\n            Return information about the current inferior.\n\n            `target_id` is ignored.\n            '
            return self._target()

        @lock_host
        def targets(self, target_ids=None):
            if False:
                return 10
            "\n            Return information about the debugger's current targets.\n\n            `target_ids` is ignored. Only the current target is returned. This\n            method is only implemented to maintain API compatibility with the\n            LLDBAdaptor.\n            "
            return [self._target()]

        @validate_target
        @lock_host
        def state(self, target_id=0):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get the state of a given target.\n            `target_id` is ignored.\n            '
            return self._state()

        @validate_busy
        @validate_target
        @lock_host
        def registers(self, target_id=0, thread_id=None, registers=[]):
            if False:
                i = 10
                return i + 15
            '\n            Get the register values for a given target/thread.\n            `target_id` is ignored.\n            '
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
                else:
                    raise UnknownArchitectureException()
            return regs

        @validate_busy
        @validate_target
        @lock_host
        def stack_pointer(self, target_id=0, thread_id=None):
            if False:
                return 10
            '\n            Get the value of the stack pointer register.\n            `target_id` is ignored.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                sp_name = self.reg_names[arch]['sp']
                sp = self.get_register(sp_name)
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
            '\n            Get the value of the program counter register.\n            `target_id` is ignored.\n            '
            arch = self.get_arch()
            if arch in self.reg_names:
                pc_name = self.reg_names[arch]['pc']
                pc = self.get_register(pc_name)
            else:
                raise UnknownArchitectureException()
            return (pc_name, pc)

        @validate_busy
        @validate_target
        @lock_host
        def memory(self, address, length, target_id=0):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get the register values for .\n            Raises `FailedToReadMemoryError` if... that happens.\n\n            `address` is the address at which to start reading\n            `length` is the number of bytes to read\n            `target_id` is ignored.\n            '
            log.debug('Reading 0x{:x} bytes of memory at 0x{:x}'.format(length, address))
            t = self._vdb.getTrace()
            try:
                return t.readMemory(address, length)
            except:
                raise FailedToReadMemoryError()

        @validate_busy
        @validate_target
        @lock_host
        def stack(self, length, target_id=0, thread_id=None):
            if False:
                while True:
                    i = 10
            '\n            Get the register values for .\n\n            `length` is the number of bytes to read\n            `target_id` is a target ID (or None for the first target)\n            `thread_id` is a thread ID (or None for the selected thread)\n            '
            (sp_name, sp) = self.stack_pointer(target_id=target_id, thread_id=thread_id)
            memory = self.memory(sp, length, target_id=target_id)
            return memory

        def _get_n_opcodes_length(self, address, count):
            if False:
                return 10
            '\n            Get the number of bytes used to represent the `n` instructions\n              at `address`.\n\n            `address` is the starting address of the sequence of instructions.\n            `count` is the number of instructions to decode.\n            '
            length = 0
            t = self._vdb.getTrace()
            arch = self._vdb.arch.getArchId()
            for i in xrange(count):
                op = t.parseOpcode(address + length, arch=arch)
                length += op.size
            return length

        @validate_busy
        @validate_target
        @lock_host
        def disassemble(self, target_id=0, address=None, count=16):
            if False:
                print('Hello World!')
            '\n            Get a disassembly of the instructions at the given address.\n\n            `address` is the address at which to disassemble. If None, the\n            current program counter is used.\n            `count` is the number of instructions to disassemble.\n            '
            if address == None:
                (pc_name, address) = self.program_counter(target_id=target_id)
            length = self._get_n_opcodes_length(address, count)
            can = envi.memcanvas.StringMemoryCanvas(self._vdb.memobj, self._vdb.symobj)
            can.renderMemory(address, length, self._vdb.opcoderend)
            return str(can)

        def _get_ascii_string(self, address, min_length=4, max_length=32):
            if False:
                print('Hello World!')
            '\n            Get the ASCII string of length at least `min_length`, but\n             not more than `max_length` of it, or raise\n             `NotAStringError` if it doesnt look like an ASCII string.\n            '
            cs = []
            for i in xrange(max_length):
                try:
                    c = self.memory(address + i, 1)[0]
                except FailedToReadMemoryError:
                    break
                if ord(c) == 0:
                    break
                elif c not in string.printable:
                    break
                else:
                    cs.append(c)
            if len(cs) >= min_length:
                return ''.join(cs)
            else:
                raise NotAStringError()

        def _get_unicode_string(self, address, min_length=4, max_length=32):
            if False:
                while True:
                    i = 10
            '\n            Get the *simple* Unicode string of length at least `min_length`\n             characters, but not more than `max_length` characters of it,\n             or raise `NotAStringError` if it doesnt look like a\n             *simple* Unicode string.\n\n            *simple* Unicode is ASCII with interspersed NULLs\n            '
            cs = []
            for i in xrange(max_length):
                try:
                    b = self.memory(address + i * 2, 2)
                except FailedToReadMemoryError:
                    break
                if ord(b[1]) != 0:
                    break
                c = b[0]
                if ord(c) == 0:
                    break
                elif c not in string.printable:
                    break
                else:
                    cs.append(c)
            if len(cs) >= min_length:
                return ''.join(cs)
            else:
                raise NotAStringError()

        @validate_busy
        @validate_target
        @lock_host
        def dereference(self, pointer, target_id=0):
            if False:
                return 10
            '\n            Recursively dereference a pointer for display\n            `target_id` is ignored.\n            '
            fmt = ('<' if self.get_byte_order() == 'little' else '>') + {2: 'H', 4: 'L', 8: 'Q'}[self.get_addr_size()]
            addr = pointer
            chain = []
            while True:
                try:
                    mem = self.memory(addr, self.get_addr_size())
                except FailedToReadMemoryError:
                    break
                except Exception as e:
                    print(e)
                    print(type(e))
                    print(e.__class__.__name__)
                    break
                log.debug('read mem: {}'.format(mem))
                (ptr,) = struct.unpack(fmt, mem)
                if ptr in chain:
                    break
                chain.append(('pointer', addr))
                addr = ptr
            (p, addr) = chain[-1]
            output = self._vdb.reprPointer(addr)
            if 'Who knows?!?!!?' not in output:
                chain.append(('symbol', output))
                log.debug('symbol context: {}'.format(str(chain[-1])))
            else:
                log.debug('no symbol context')
                try:
                    chain.append(('string', self._get_ascii_string(addr)))
                except NotAStringError:
                    try:
                        chain.append(('string', self._get_unicode_string(addr)))
                    except NotAStringError:
                        pass
            log.debug('chain: {}'.format(chain))
            return chain

        @lock_host
        def command(self, command=None):
            if False:
                print('Hello World!')
            '\n            Execute a command in the debugger.\n\n            `command` is the command string to execute.\n            '
            if command:
                oldcan = self._vdb.canvas
                newcan = envi.memcanvas.StringMemoryCanvas(self._vdb.memobj, self._vdb.symobj)
                try:
                    self._vdb.canvas = newcan
                    self._vdb.onecmd(command)
                finally:
                    self._vdb.canvas = oldcan
                return str(newcan).rstrip('\n')
            else:
                raise Exception('No command specified')
            return res

        @lock_host
        def disassembly_flavor(self):
            if False:
                return 10
            "\n            Return the disassembly flavor setting for the debugger.\n\n            Returns 'intel' or 'att'\n            "
            return 'intel'

        def capabilities(self):
            if False:
                while True:
                    i = 10
            "\n            Return a list of the debugger's capabilities.\n\n            Thus far only the 'async' capability is supported. This indicates\n            that the debugger host can be queried from a background thread,\n            and that views can use non-blocking API requests without queueing\n            requests to be dispatched next time the debugger stops.\n            "
            return ['async']

        def _state(self):
            if False:
                i = 10
                return i + 15
            '\n            Get the state of a given target. Internal use.\n            '
            if not self._vdb.getTrace().isAttached():
                state = 'invalid'
            elif self._vdb.getTrace().isRunning():
                state = 'running'
            else:
                state = 'stopped'
            return state

        def get_registers(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._vdb.getTrace().getRegisters()

        def get_register(self, reg_name):
            if False:
                for i in range(10):
                    print('nop')
            return self.get_registers()[reg_name]

        def get_registers_x86_64(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.get_registers()

        def get_registers_x86(self):
            if False:
                while True:
                    i = 10
            return self.get_registers()

        def get_registers_arm(self):
            if False:
                print('Hello World!')
            return self.get_registers()

        def get_registers_sse(self, num=8):
            if False:
                while True:
                    i = 10
            sse = {}
            for (k, v) in self.get_registers().items():
                if k.startswith('xmm'):
                    sse[k] = v
            return sse

        def get_registers_fpu(self):
            if False:
                while True:
                    i = 10
            fpu = {}
            for (k, v) in self.get_registers().items():
                if k.startswith('st'):
                    fpu[k] = v
            return fpu

        def get_next_instruction(self):
            if False:
                print('Hello World!')
            dis = self.disassemble(address=self.program_counter()[1], count=1)
            return dis.partition('\n')[0].strip()

        def get_arch(self):
            if False:
                i = 10
                return i + 15
            arch = self._vdb.getTrace().getMeta('Architecture')
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
            return 'little'

    class VDBCommand(DebuggerCommand, vtrace.Notifier):
        """
        Debugger command class for VDB
        """

        def __init__(self, host):
            if False:
                i = 10
                return i + 15
            '\n            vdb is the debugger instance\n            vtrace is the vtrace module?\n            '
            super(VDBCommand, self).__init__()
            self._vdb = host
            self._vtrace = vtrace
            self.register_hooks()

        def invoke(self, arg, from_tty):
            if False:
                while True:
                    i = 10
            self.handle_command(arg)

        def register_hooks(self):
            if False:
                for i in range(10):
                    print('nop')
            self._vdb.registerNotifier(vtrace.NOTIFY_ALL, self)

        def unregister_hooks(self):
            if False:
                return 10
            self._vdb.deregisterNotifier(vtrace.NOTIFY_ALL, self)

        def notify(self, event, trace):
            if False:
                return 10
            if event == self._vtrace.NOTIFY_DETACH:
                self.exit_handler(event)
            elif event == self._vtrace.NOTIFY_EXIT:
                self.exit_handler(event)
            elif event == self._vtrace.NOTIFY_BREAK:
                self.stop_handler(event)
            elif event == self._vtrace.NOTIFY_STEP:
                self.stop_handler(event)
            elif event == self._vtrace.NOTIFY_CONTINUE:
                self.cont_handler(event)

        def stop_handler(self, event):
            if False:
                i = 10
                return i + 15
            self.adaptor.update_state()
            voltron.server.dispatch_queue()
            log.debug('Inferior stopped')

        def exit_handler(self, event):
            if False:
                while True:
                    i = 10
            log.debug('Inferior exited')
            voltron.server.cancel_queue()
            voltron.server.stop()

        def cont_handler(self, event):
            if False:
                print('Hello World!')
            log.debug('Inferior continued')

    class VDBAdaptorPlugin(DebuggerAdaptorPlugin):
        host = 'vdb'
        adaptor_class = VDBAdaptor
        command_class = VDBCommand