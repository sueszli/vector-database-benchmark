from __future__ import print_function
import os
import struct
import logging
import threading
import codecs
from collections import namedtuple
from voltron.api import NoSuchThreadException
from voltron.colour import uncolour
from voltron.plugin import voltron
from voltron.dbg import lock_host, validate_target, validate_busy, DebuggerAdaptor, InvalidPointerError, DebuggerCommand, DebuggerAdaptorPlugin
from voltron.plugins.debugger import xrange
try:
    import lldb
    HAVE_LLDB = True
except ImportError:
    HAVE_LLDB = False
log = logging.getLogger('debugger')
MAX_DEREF = 16
if HAVE_LLDB:

    class LLDBAdaptor(DebuggerAdaptor):
        """
        The interface with an instance of LLDB
        """

        def __init__(self, host=None):
            if False:
                print('Hello World!')
            self.listeners = []
            self.host_lock = threading.RLock()
            if host:
                log.debug('Passed a debugger host')
                self.host = host
            elif lldb.debugger:
                log.debug('lldb.debugger is valid - probably running inside LLDB')
                self.host = lldb.debugger
            else:
                log.debug('No debugger host found - creating one')
                self.host = lldb.SBDebugger.Create()
                self.host.SetAsync(False)

        @property
        def host(self):
            if False:
                return 10
            '\n            Get the debugger host object that this adaptor talks to. Used by\n            custom API plugins to talk directly to the debugger.\n            '
            return self._host

        @host.setter
        def host(self, value):
            if False:
                for i in range(10):
                    print('nop')
            self._host = value

        def normalize_triple(self, triple):
            if False:
                i = 10
                return i + 15
            "\n            Returns a (cpu, platform, abi) triple\n\n            Returns None for any fields that can't be elided\n            "
            s = triple.split('-')
            (arch, platform, abi) = (s[0], s[1], '-'.join(s[2:]))
            if arch == 'x86_64h':
                arch = 'x86_64'
            return (arch, platform, abi)

        def version(self):
            if False:
                i = 10
                return i + 15
            "\n            Get the debugger's version.\n\n            Returns a string containing the debugger's version\n            (e.g. 'lldb-310.2.37')\n            "
            return self.host.GetVersionString()

        def _target(self, target_id=0):
            if False:
                print('Hello World!')
            '\n            Return information about the specified target.\n\n            Returns data in the following structure:\n            {\n                "id":       0,         # ID that can be used in other funcs\n                "file":     "/bin/ls", # target\'s binary file\n                "arch":     "x86_64",  # target\'s architecture\n                "state:     "stopped"  # state\n            }\n            '
            t = self.host.GetTargetAtIndex(target_id)
            d = {}
            d['id'] = target_id
            d['state'] = self.host.StateAsCString(t.process.GetState())
            d['file'] = t.GetExecutable().fullpath
            try:
                (d['arch'], _, _) = self.normalize_triple(t.triple)
            except:
                d['arch'] = None
            if d['arch'] == 'i386':
                d['arch'] = 'x86'
            d['byte_order'] = 'little' if t.byte_order == lldb.eByteOrderLittle else 'big'
            d['addr_size'] = t.addr_size
            return d

        @lock_host
        def target(self, target_id=0):
            if False:
                while True:
                    i = 10
            '\n            Return information about the specified target.\n            '
            return self._target(target_id=target_id)

        @lock_host
        def targets(self, target_ids=None):
            if False:
                print('Hello World!')
            '\n            Return information about the debugger\'s current targets.\n\n            `target_ids` is an array of target IDs (or None for all targets)\n\n            Returns data in the following structure:\n            [\n                {\n                    "id":       0,         # ID that can be used in other funcs\n                    "file":     "/bin/ls", # target\'s binary file\n                    "arch":     "x86_64",  # target\'s architecture\n                    "state:     "stopped"  # state\n                }\n            ]\n            '
            targets = []
            if not target_ids:
                n = self.host.GetNumTargets()
                target_ids = range(n)
            log.debug('Getting info for {} targets'.format(len(target_ids)))
            for i in target_ids:
                targets.append(self._target(i))
            return targets

        @validate_target
        @lock_host
        def state(self, target_id=0):
            if False:
                return 10
            '\n            Get the state of a given target.\n\n            `target_id` is a target ID (or None for the first target)\n            '
            target = self.host.GetTargetAtIndex(target_id)
            state = self.host.StateAsCString(target.process.GetState())
            return state

        @validate_busy
        @validate_target
        @lock_host
        def registers(self, target_id=0, thread_id=None, registers=[]):
            if False:
                return 10
            '\n            Get the register values for a given target/thread.\n\n            `target_id` is a target ID (or None for the first target)\n            `thread_id` is a thread ID (or None for the selected thread)\n            '
            target = self.host.GetTargetAtIndex(target_id)
            t_info = self._target(target_id)
            if not thread_id:
                thread_id = target.process.selected_thread.id
            try:
                thread = target.process.GetThreadByID(thread_id)
            except:
                raise NoSuchThreadException()
            if t_info['arch'] in self.reg_names:
                if 'pc' in registers:
                    registers.remove('pc')
                    registers.append(self.reg_names[t_info['arch']]['pc'])
                if 'sp' in registers:
                    registers.remove('sp')
                    registers.append(self.reg_names[t_info['arch']]['sp'])
            else:
                raise Exception('Unsupported architecture: {}'.format(t_info['arch']))
            regs = thread.GetFrameAtIndex(0).GetRegisters()
            objs = []
            for i in xrange(len(regs)):
                objs += regs[i]
            regs = {}
            for reg in objs:
                val = 'n/a'
                if reg.value is not None:
                    try:
                        val = reg.GetValueAsUnsigned()
                    except:
                        reg = None
                elif reg.num_children > 0:
                    try:
                        children = []
                        for i in xrange(reg.GetNumChildren()):
                            children.append(int(reg.GetChildAtIndex(i, lldb.eNoDynamicValues, True).value, 16))
                        if t_info['byte_order'] == 'big':
                            children = list(reversed(children))
                        val = int(codecs.encode(struct.pack('{}B'.format(len(children)), *children), 'hex'), 16)
                    except:
                        pass
                if registers == [] or reg.name in registers:
                    regs[reg.name] = val
            return regs

        @validate_busy
        @validate_target
        @lock_host
        def stack_pointer(self, target_id=0, thread_id=None):
            if False:
                return 10
            '\n            Get the value of the stack pointer register.\n\n            `target_id` is a target ID (or None for the first target)\n            `thread_id` is a thread ID (or None for the selected thread)\n            '
            regs = self.registers(target_id=target_id, thread_id=thread_id)
            target = self._target(target_id=target_id)
            if target['arch'] in self.reg_names:
                sp_name = self.reg_names[target['arch']]['sp']
                sp = regs[sp_name]
            else:
                raise Exception('Unsupported architecture: {}'.format(target['arch']))
            return (sp_name, sp)

        @validate_busy
        @validate_target
        @lock_host
        def program_counter(self, target_id=0, thread_id=None):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get the value of the program counter register.\n\n            `target_id` is a target ID (or None for the first target)\n            `thread_id` is a thread ID (or None for the selected thread)\n            '
            regs = self.registers(target_id=target_id, thread_id=thread_id)
            target = self._target(target_id=target_id)
            if target['arch'] in self.reg_names:
                pc_name = self.reg_names[target['arch']]['pc']
                pc = regs[pc_name]
            else:
                raise Exception('Unsupported architecture: {}'.format(target['arch']))
            return (pc_name, pc)

        @validate_busy
        @validate_target
        @lock_host
        def memory(self, address, length, target_id=0):
            if False:
                return 10
            '\n            Get the register values for .\n\n            `address` is the address at which to start reading\n            `length` is the number of bytes to read\n            `target_id` is a target ID (or None for the first target)\n            '
            target = self.host.GetTargetAtIndex(target_id)
            log.debug('Reading 0x{:x} bytes of memory at 0x{:x}'.format(length, address))
            error = lldb.SBError()
            memory = target.process.ReadMemory(address, length, error)
            if not error.Success():
                raise Exception('Failed reading memory: {}'.format(error.GetCString()))
            return memory

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

        @validate_busy
        @validate_target
        @lock_host
        def disassemble(self, target_id=0, address=None, count=None):
            if False:
                print('Hello World!')
            '\n            Get a disassembly of the instructions at the given address.\n\n            `address` is the address at which to disassemble. If None, the\n            current program counter is used.\n            `count` is the number of instructions to disassemble.\n            '
            if address is None:
                (pc_name, address) = self.program_counter(target_id=target_id)
            output = self.command('disassemble -s {} -c {}'.format(address, count))
            output = uncolour(output)
            return output

        @validate_busy
        @validate_target
        @lock_host
        def dereference(self, pointer, target_id=0):
            if False:
                print('Hello World!')
            '\n            Recursively dereference a pointer for display\n            '
            t = self.host.GetTargetAtIndex(target_id)
            error = lldb.SBError()
            addr = pointer
            chain = []
            for i in range(0, MAX_DEREF):
                ptr = t.process.ReadPointerFromMemory(addr, error)
                if error.Success():
                    if ptr in chain:
                        chain.append(('circular', 'circular'))
                        break
                    chain.append(('pointer', addr))
                    addr = ptr
                else:
                    break
            if len(chain) == 0:
                raise InvalidPointerError('0x{:X} is not a valid pointer'.format(pointer))
            (p, addr) = chain[-1]
            sbaddr = lldb.SBAddress(addr, t)
            ctx = t.ResolveSymbolContextForAddress(sbaddr, lldb.eSymbolContextEverything)
            if ctx.IsValid() and ctx.GetSymbol().IsValid():
                fstart = ctx.GetSymbol().GetStartAddress().GetLoadAddress(t)
                offset = addr - fstart
                chain.append(('symbol', '{} + 0x{:X}'.format(ctx.GetSymbol().name, offset)))
                log.debug('symbol context: {}'.format(str(chain[-1])))
            else:
                log.debug('no symbol context')
                s = t.process.ReadCStringFromMemory(addr, 256, error)
                for i in range(0, len(s)):
                    if ord(s[i]) >= 128:
                        s = s[:i]
                        break
                if len(s):
                    chain.append(('string', s))
            return chain

        @lock_host
        def command(self, command=None):
            if False:
                while True:
                    i = 10
            '\n            Execute a command in the debugger.\n\n            `command` is the command string to execute.\n            '
            if command:
                res = lldb.SBCommandReturnObject()
                ci = self.host.GetCommandInterpreter()
                ci.HandleCommand(str(command), res, False)
                if res.Succeeded():
                    output = res.GetOutput()
                    return output.strip() if output else ''
                else:
                    raise Exception(res.GetError().strip())
            else:
                raise Exception('No command specified')

        @lock_host
        def disassembly_flavor(self):
            if False:
                return 10
            "\n            Return the disassembly flavor setting for the debugger.\n\n            Returns 'intel' or 'att'\n            "
            res = lldb.SBCommandReturnObject()
            ci = self.host.GetCommandInterpreter()
            ci.HandleCommand('settings show target.x86-disassembly-flavor', res)
            if res.Succeeded():
                output = res.GetOutput().strip()
                flavor = output.split()[-1]
                if flavor == 'default':
                    flavor = 'att'
            else:
                raise Exception(res.GetError().strip())
            return flavor

        @validate_busy
        @validate_target
        @lock_host
        def breakpoints(self, target_id=0):
            if False:
                return 10
            '\n            Return a list of breakpoints.\n\n            Returns data in the following structure:\n            [\n                {\n                    "id":           1,\n                    "enabled":      True,\n                    "one_shot":     False,\n                    "hit_count":    5,\n                    "locations": [\n                        {\n                            "address":  0x100000cf0,\n                            "name":     \'main\'\n                        }\n                    ]\n                }\n            ]\n            '
            breakpoints = []
            t = self.host.GetTargetAtIndex(target_id)
            s = lldb.SBStream()
            for i in range(0, t.GetNumBreakpoints()):
                b = t.GetBreakpointAtIndex(i)
                locations = []
                for j in range(0, b.GetNumLocations()):
                    loc = b.GetLocationAtIndex(j)
                    s.Clear()
                    loc.GetAddress().GetDescription(s)
                    desc = s.GetData()
                    locations.append({'address': loc.GetLoadAddress(), 'name': desc})
                breakpoints.append({'id': b.id, 'enabled': b.enabled, 'one_shot': b.one_shot, 'hit_count': b.GetHitCount(), 'locations': locations})
            return breakpoints

        @validate_busy
        @validate_target
        @lock_host
        def backtrace(self, target_id=0, thread_id=None):
            if False:
                while True:
                    i = 10
            '\n            Return a list of stack frames.\n            '
            target = self.host.GetTargetAtIndex(target_id)
            if not thread_id:
                thread_id = target.process.selected_thread.id
            try:
                thread = target.process.GetThreadByID(thread_id)
            except:
                raise NoSuchThreadException()
            frames = []
            for frame in thread:
                start_addr = frame.GetSymbol().GetStartAddress().GetFileAddress()
                offset = frame.addr.GetFileAddress() - start_addr
                ctx = frame.GetSymbolContext(lldb.eSymbolContextEverything)
                mod = ctx.GetModule()
                name = '{mod}`{symbol} + {offset}'.format(mod=os.path.basename(str(mod.file)), symbol=frame.name, offset=offset)
                frames.append({'index': frame.idx, 'addr': frame.addr.GetFileAddress(), 'name': name})
            return frames

        def capabilities(self):
            if False:
                while True:
                    i = 10
            "\n            Return a list of the debugger's capabilities.\n\n            Thus far only the 'async' capability is supported. This indicates\n            that the debugger host can be queried from a background thread,\n            and that views can use non-blocking API requests without queueing\n            requests to be dispatched next time the debugger stops.\n            "
            return ['async']

        def register_command_plugin(self, name, cls):
            if False:
                return 10
            '\n            Register a command plugin with the LLDB adaptor.\n            '
            if not voltron.commands:
                voltron.commands = namedtuple('VoltronCommands', [])

            def create_invocation(obj):
                if False:
                    print('Hello World!')

                @staticmethod
                def invoker(debugger, command, result, env_dict):
                    if False:
                        for i in range(10):
                            print('nop')
                    obj.invoke(*command.split())
                return invoker
            setattr(voltron.commands, name, create_invocation(cls()))
            self.host.HandleCommand('command script add -f voltron.commands.{} {}'.format(name, name))

    class LLDBCommand(DebuggerCommand):
        """
        Debugger command class for LLDB
        """

        @staticmethod
        def _invoke(debugger, command, *args):
            if False:
                i = 10
                return i + 15
            voltron.command.handle_command(command)

        def __init__(self):
            if False:
                return 10
            super(LLDBCommand, self).__init__()
            self.hook_idx = None
            self.adaptor = voltron.debugger
            self.adaptor.command('script import voltron')
            self.adaptor.command('command script add -f entry.invoke voltron')
            self.register_hooks(True)

        def invoke(self, debugger, command, result, dict):
            if False:
                for i in range(10):
                    print('nop')
            self.handle_command(command)

        def register_hooks(self, quiet=False):
            if False:
                return 10
            try:
                output = self.adaptor.command('target stop-hook list')
                if 'voltron' not in output:
                    output = self.adaptor.command("target stop-hook add -o 'voltron stopped'")
                    try:
                        log.debug('Saving hook index for unregistering.')
                        self.hook_idx = int(output.split()[2][1:])
                    except Exception as e:
                        log.warning(f'Exception when saving hook index for unregistering. {e}')
                        pass
                self.registered = True
                if not quiet:
                    print('Registered stop-hook')
            except:
                if not quiet:
                    print('No targets')

        def unregister_hooks(self):
            if False:
                print('Hello World!')
            self.adaptor.command('target stop-hook delete {}'.format(self.hook_idx if self.hook_idx else ''))
            self.registered = False

    class LLDBAdaptorPlugin(DebuggerAdaptorPlugin):
        host = 'lldb'
        adaptor_class = LLDBAdaptor
        command_class = LLDBCommand