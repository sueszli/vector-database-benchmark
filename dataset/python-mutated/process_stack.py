"""
@author:       Edwin Smulders
@license:      GNU General Public License 2.0 or later
@contact:      mail@edwinsmulders.eu
"""
import volatility.plugins.linux.process_info as linux_process_info
import volatility.plugins.linux.check_syscall as linux_check_syscall
import volatility.plugins.linux.common as linux_common
import volatility.debug as debug
import struct
import os.path
verbose_stack_arguments = True
stats = {}
stats['tasks'] = 0
stats['threads'] = 0
stats['tasks_ignored'] = 0
stats['tasks_zero_frames'] = 0
stats['threads_zero_frames'] = 0
stats['libc_start'] = 0
stats['main'] = 0
stats['frames'] = {}
stats['frames']['possible_frames'] = 0
stats['frames']['function_address'] = 0
stats['frames']['symbols'] = 0
try:
    import distorm3
    distorm_loaded = True
except:
    distorm_loaded = False
try:
    import elftools
    elftools_loaded = True
except:
    elftools_loaded = False

def yield_address(space, start, length=None, reverse=False):
    if False:
        i = 10
        return i + 15
    '\n    A function to read a series of values starting at a certain address.\n\n    @param space: address space\n    @param start: starting address\n    @param length: the size of the values to read\n    @param reverse: option to read in the other direction\n    @return: an iterator\n    '
    if not length:
        length = linux_process_info.address_size
    cont = True
    while space.is_valid_address(start) and cont:
        try:
            value = read_address(space, start, length)
            yield value
        except struct.error:
            cont = False
            yield None
        if reverse:
            start -= length
        else:
            start += length

def read_address(space, start, length=None):
    if False:
        i = 10
        return i + 15
    '\n    Read an address in a space, at a location, of a certain length.\n    @param space: the address space\n    @param start: the address\n    @param length: size of the value\n    '
    if not length:
        length = linux_process_info.address_size
    fmt = '<I' if length == 4 else '<Q'
    return struct.unpack(fmt, space.read(start, length))[0]

class linux_process_stack(linux_process_info.linux_process_info):
    """
    Plugin to do analysis on the stack of user space applications.
    """

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        linux_process_info.linux_process_info.__init__(self, config, *args, **kwargs)
        self._config.add_option('SYMBOL-DIR', short_option='s', default=None, help='Directory containing files with function symbols', type='str')
        self._config.add_option('DUMP-FILE', short_option='o', default=None, help='Dump an annotated stack to this file', type='str')
        self.symbols = None
        self.undefined = None
        self.dump_file = None
        if distorm_loaded:
            self.decode_as = distorm3.Decode32Bits if linux_process_info.address_size == 4 else distorm3.Decode64Bits
        else:
            debug.error('You really need the distorm3 python module for this plugin to function properly.')

    def load_symbols(self, dir):
        if False:
            return 10
        '\n        Loads function symbols from a directory.\n        @param dir: the directory\n        @return: a symbol dict or None\n        '
        if os.path.isdir(dir):
            debug.info('Loading function symbols from directory: {}'.format(dir))
            symbols = {}
            for filename in os.listdir(dir):
                if filename[-7:] == '.dynsym':
                    libname = filename[:-7]
                elif filename[-8:] == '.symbols':
                    libname = filename[:-8]
                else:
                    libname = filename
                if not libname in symbols:
                    symbols[libname] = {}
                with open(os.path.join(dir, filename), 'r') as f:
                    for line in f:
                        line = line.strip().split(' ')
                        if len(line) == 2:
                            pass
                        else:
                            offset = int(line[0], 16)
                            t = line[1]
                            name = line[2]
                            symbols[libname][offset] = name
            return symbols
        else:
            debug.warning('Loading function symbols from directory: Not a valid directory: {}'.format(dir))
        return None

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        lpi = linux_process_info
        if self._config.SYMBOL_DIR:
            self.symbols = self.load_symbols(self._config.SYMBOL_DIR)
        if self._config.DUMP_FILE:
            try:
                self.dump_file = open(self._config.DUMP_FILE, 'a+')
                debug.info('Opened {} for writing'.format(self._config.DUMP_FILE))
            except IOError:
                debug.error('Failed to open %s for writing'.format(self._config.DUMP_FILE))
        for p in linux_process_info.linux_process_info.calculate(self):
            stats['tasks'] += 1
            if p:
                for (i, task) in enumerate(p.threads):
                    stats['threads'] += 1
                    debug.info('Starting analysis of task: pid {}, thread name {}'.format(task.pid, task.comm))
                    debug.info('=================================================')
                    yield self.analyze_stack(p, task, i)
            else:
                stats['tasks_ignored'] += 1

    def analyze_stack(self, process_info, task, thread_number):
        if False:
            print('Hello World!')
        '\n        Analyzes the stack, building the stack frames and performing validation\n        @param process_info: The porcess info object\n        @param task: the task_struct\n        @param thread_number: the thread number for use in process info\n        @return: a tuple (process info, registers, frames list) or None\n        '
        p = process_info
        i = thread_number
        is_thread = i != 0
        for (low, high) in p.thread_stack_ranges:
            if low <= p.thread_registers[i].rsp <= high:
                debug.info('Found the stack at 0x{:016x}-0x{:016x}'.format(low, high))
                stack_low = low
                stack_high = high
        address_size = linux_process_info.address_size
        (libc_start, libc_end, libc_name) = process_info.get_map_by_name('libc-', 'r-x')
        debug.info('Found libc ({}) at range: 0x{:016x}-0x{:016x}'.format(libc_name, libc_start, libc_end))
        debug.info('Program code located at 0x{:016x}-0x{:016x}'.format(p.mm_start_code, p.mm_end_code))
        if is_thread:
            debug.info("Current task is a thread, we don't expect to find the start/main return addresses!")
        entry_point = self.find_entry_point(p.proc_as, p.mm_start_code)
        offset = p.mm.arg_start % address_size
        stack_arguments = p.mm.arg_start - address_size - offset
        libc_start_main_stack_frame = None
        main_scan_start = None
        if not is_thread and entry_point:
            debug.info("Executable entry point ('_start' function): 0x{:016x}".format(entry_point))
            return_start = entry_point + 41
            debug.info('Scanning for return address of __libc_start_main function, starting at program arguments (0x{:016x}) downwards'.format(stack_arguments))
            return_libc_start = self.find_return_libc_start(p.proc_as, stack_arguments, return_start)
            if return_libc_start:
                stats['libc_start'] += 1
                debug.info('Found the __libc_start_main return address (0x{:016x}) at address 0x{:016x}'.format(return_start, return_libc_start))
                debug.info('Scanning for return address of main function, starting at __libc_start_main return address (0x{:016x}) downwards'.format(return_libc_start))
                main_scan_start = return_libc_start
                libc_start_main_stack_frame = stack_frame(return_libc_start + address_size, p.proc_as, 0)
        if not main_scan_start:
            if not is_thread:
                main_scan_start = stack_arguments
                debug.info('Scanning for return address of main function, starting at program arguments (0x{:016x}) downwards'.format(main_scan_start))
            else:
                main_scan_start = stack_high
        found_main = self.find_return_main(process_info.proc_as, libc_start, libc_end, main_scan_start)
        if found_main:
            stats['main'] += 1
            (stack_main, main_offset) = found_main
            debug.info('Found main stackframe at 0x{:016x}'.format(stack_main))
            main_frame = stack_frame(stack_main + address_size, p.proc_as, -1)
            main_pointer = main_frame.ret_address + main_offset + address_size
            main_address = read_address(p.proc_as, main_pointer, address_size)
            debug.info('The address of the main function is 0x{:016x}'.format(main_address))
            main_frame.function = main_address
        else:
            debug.warning('Unable to find address of main stackframe')
            debug.info('Assuming no frame pointers')
            main_address = 0
            main_frame = None
        frames = []
        st = None
        if self.has_frame_pointer(main_address, p.proc_as):
            debug.info('Register %rbp was not 0, trying old-school stack frames')
            frames += self.find_oldschool_frames(p, p.proc_as, p.thread_registers[i])
        elif found_main:
            debug.info('No old-school stack frames detected, scanning for return addresses')
            address = p.thread_registers[i].rsp
            end = main_frame.ret_address - address_size
            frames += self.find_scanned_frames(p, address, end)
            main_frame.frame_number = frames[-1].frame_number + 1
            frames.append(main_frame)
        else:
            address = p.thread_registers[i].rsp
            end = stack_high
            frames += self.find_scanned_frames(p, address, end)
        if len(frames) > 0:
            lastframe = frames[-1]
            while lastframe.ebp and p.is_thread_stack_pointer(lastframe.ebp) and (not lastframe.ebp == lastframe.ebp_address):
                newframe = stack_frame(lastframe.ebp + address_size * 2, p.proc_as, lastframe.frame_number + 1)
                frames.append(newframe)
                lastframe = newframe
            if main_frame:
                if main_frame.address == lastframe.address:
                    lastframe.function = main_frame.function
                else:
                    frames.append(main_frame)
            if libc_start_main_stack_frame:
                if lastframe.address != libc_start_main_stack_frame.address:
                    frames.append(libc_start_main_stack_frame)
        else:
            if main_frame:
                frames.append(main_frame)
            if libc_start_main_stack_frame:
                frames.append(libc_start_main_stack_frame)
        for frame in frames:
            if not frame.function:
                frame.function = self.find_function_address(p.proc_as, frame.ret)
            frame.symbol = self.find_function_symbol(task, frame.function)
            stats['frames']['possible_frames'] += 1
            if frame.function:
                stats['frames']['function_address'] += 1
            if frame.symbol:
                stats['frames']['symbols'] += 1
        if len(frames) == 0:
            if is_thread:
                stats['threads_zero_frames'] += 1
            else:
                stats['tasks_zero_frames'] += 1
        return (p, p.thread_registers[i], frames)

    def find_oldschool_frames(self, p, proc_as, registers):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function builds a list of stack frames using the old frame pointer\n        @param p: process info\n        @param proc_as: process address space\n        @param registers: cpu registers\n        @return: a list of frames\n        '
        frames = []
        address_size = linux_process_info.address_size
        rbp = registers.rbp
        rsp_value = read_address(proc_as, registers.rsp)
        frame_number = 1
        st = stack_frame(rbp + 16, proc_as, frame_number)
        address = registers.rsp
        frame0_addr = 0
        foundframe0 = False
        frame0 = None
        while address < st.ebp_address:
            value = read_address(p.proc_as, address)
            if value == st.ebp_address:
                frame0_addr = address + address_size * 2
                foundframe0 = True
                break
            address += address_size
        if frame0_addr == 0 and p.is_code_pointer(rsp_value):
            frame0_addr = registers.rsp + address_size
            foundframe0 = True
        if not foundframe0:
            st.frame_number = 0
        else:
            frame0 = stack_frame(frame0_addr, p.proc_as, 0)
        if frame0:
            frames.append(frame0)
        frames.append(st)
        return frames

    def find_scanned_frames(self, p, address, end):
        if False:
            i = 10
            return i + 15
        '\n        Find frames by scanning for return addresses.\n        @param p: process info object\n        @param address: Start address\n        @param end: End address\n        @return: a list of frames\n        '
        address_size = linux_process_info.address_size
        frames = []
        debug.info('Scan range (%rsp to end) = (0x{:016x} to 0x{:016x})'.format(address, end))
        count = 0
        while address <= end:
            if p.proc_as.is_valid_address(address) and self.is_return_address(read_address(p.proc_as, address, address_size), p):
                st = stack_frame(address + address_size, p.proc_as, count)
                frames.append(st)
                count += 1
            address += address_size
        return frames

    def find_entry_point(self, proc_as, start_code):
        if False:
            return 10
        '\n        Read the entry point from the program header.\n        @param proc_as: Process address space\n        @param start_code: Start of the program code mapping\n        @return The address of the entry point (_start)\n        '
        if not proc_as.is_valid_address(start_code + 24):
            debug.info('We could not find program entry point, skipping _start detection')
            return False
        offset = read_address(proc_as, start_code + 24)
        if offset > start_code:
            return offset
        else:
            return start_code + offset

    def validate_stack_frames(self, frames):
        if False:
            print('Hello World!')
        '\n        Attempt to validate stackframes, broken and unused.\n        @param frames: list of frames\n        @return: None\n        '
        prev_function = 0
        to_remove = []
        for frame in frames[::-1]:
            if prev_function < frame.ret:
                prev_function = frame.function
            else:
                frames.remove(frame)

    def is_return_address(self, address, process_info):
        if False:
            print('Hello World!')
        "\n        Checks if the address is a return address by checking if the preceding instruction is a 'CALL'.\n        @param address: An address\n        @param process_info: process info object\n        @return True or False\n        "
        proc_as = process_info.proc_as
        size = 5
        if distorm_loaded and process_info.is_code_pointer(address):
            offset = address - size
            instr = distorm3.Decode(offset, proc_as.read(offset, size), self.decode_as)
            if len(instr) > 0:
                return instr[-1][2][:4] == 'CALL'
        return False

    def find_return_libc_start(self, proc_as, start_stack, return_start):
        if False:
            print('Hello World!')
        '\n        Scans the stack for a certain address, in this case the return address of __libc_start_main.\n        @param proc_as: Process address space\n        @param start_stack: Start address to search\n        @param return_start: The return address to find\n        @return The address found or None\n        '
        address = start_stack
        for value in yield_address(proc_as, start_stack, reverse=True):
            if value == return_start:
                debug.info('Scanned {} stack addresses before finding the __libc_start_main return address'.format((start_stack - address) / linux_process_info.address_size))
                return address
            address -= linux_process_info.address_size
        debug.info('Exhausted search for __libc_start_main return address at stack address {:016x}'.format(address))
        return None

    def find_return_main(self, proc_as, libc_start, libc_end, start_address):
        if False:
            while True:
                i = 10
        '\n        Find the return address of the main function by scanning for pointers into libc. At this point we will look\n        for specific patterns in the code, to gather addresses.\n        @param proc_as: Process address space\n        @param libc_start: Start address of libc code\n        @param libc_end: End address of libc code\n        @param start_address: The address to start the scan at.\n        @return: The address on the stack and an offset (the location of the main address on the stack) or None/False\n        '
        if not distorm_loaded:
            return

        def is_return_address(address):
            if False:
                print('Hello World!')
            size = 4
            bytestr = proc_as.read(address - size, size)
            single_instr = distorm3.Decode(address - size, bytestr, self.decode_as)
            if len(single_instr) == 1 and single_instr[0][2][:4] == 'CALL':
                part = single_instr[0][2].split('[')[1]
                if part[:4] == 'RSP+':
                    rspoffset = int(part.split('+')[1][2:-1], 16)
                    return rspoffset
            size = 10
            bytestr = proc_as.read(address - size, size)
            possible = ['RCX', 'RAX']
            instr = distorm3.Decode(address - size, bytestr, self.decode_as)
            checkother = False
            if 0 < len(instr) < 3:
                pass
            elif len(instr) == 3:
                checkother = True
            else:
                return False
            last_instr = instr[-1][2].split(' ')
            register = None
            if last_instr[0] == 'CALL' and last_instr[1] in possible:
                register = last_instr[1]
            else:
                return None
            if checkother:
                mov = 'MOV ' + register
                confirmed = True
                movinstr = None
                saveinstr = None
                if mov in instr[0][2]:
                    movinstr = instr[0][2]
                    saveinstr = instr[1][2]
                elif mov in instr[1][2]:
                    saveinstr = instr[0][2]
                    movinstr = instr[1][2]
                else:
                    confirmed = False
                if movinstr != None:
                    part = movinstr.split('[')[1]
                    if part[:4] == 'RSP+':
                        rspoffset = int(part.split('+')[1][2:-1], 16)
                        return rspoffset
            return False
        addr = start_address
        counter = 0
        invalid = 0
        for value in yield_address(proc_as, start_address, reverse=True):
            if libc_start <= value <= libc_end:
                counter += 1
                if not proc_as.is_valid_address(value):
                    invalid += 1
                else:
                    retval = is_return_address(value)
                    if retval:
                        debug.info('Scanned {} libc addresses on the stack before finding the main return address'.format(counter))
                        return (addr, retval)
            addr -= linux_process_info.address_size
        debug.info('Scanned {} libc addresses on the stack, did not find the main return address'.format(counter))
        debug.info('Of these addresses, {} were invalid (e.g. due to swap)'.format(invalid))

    def find_locals_size(self, proc_as, frames):
        if False:
            while True:
                i = 10
        "\n        Find the size of the locals of the function, similar to GDB's prologue analysis.\n        Buggy and not actually used.\n\n        @param proc_as: Process address space\n        @param frames: a list of stack frames\n        @return None\n        "
        if not distorm_loaded:
            return
        for frame in frames:
            if frame.function:
                instr = distorm3.Decode(frame.function, proc_as.read(frame.function, 8), self.decode_as)
                if self.is_function_header(instr) and len(instr) > 2:
                    test = instr[2][2].split(' ')
                    if test[0] == 'SUB' and test[1] == 'RSP,':
                        frame.locals_size = int(test[2][2:], 16)

    def has_frame_pointer(self, function_address, proc_as):
        if False:
            i = 10
            return i + 15
        '\n        Check if the function at function_address has a frame pointer.\n        @param function_address: An address of a function (code)\n        @param proc_as: Process address space\n        @return: True or False\n        '
        return proc_as.read(function_address, 1) == 'U'

    def is_function_header(self, instructions):
        if False:
            i = 10
            return i + 15
        '\n        Check if something is a function header (with frame pointer and locals).\n        @param instructions: distorm disassembled instructions\n        @return True or False\n        '
        return len(instructions) > 1 and instructions[0][2] == 'PUSH RBP' and (instructions[1][2] == 'MOV RBP, RSP')

    def find_function_symbol(self, task, address):
        if False:
            while True:
                i = 10
        '\n        Match a function symbol to a functiona address.\n        @param task: the task_struct\n        @param address:  The function address\n        @return: The function symbol or None\n        '
        if self.symbols:
            for vma in task.get_proc_maps():
                if vma.vm_start <= address <= vma.vm_end:
                    lib = linux_common.get_path(task, vma.vm_file)
                    offset = address - vma.vm_start
                    if type(lib) == list:
                        lib = ''
                    base = os.path.basename(lib)
                    if base in self.symbols:
                        if offset in self.symbols[base]:
                            debug.info('Instruction was a call to 0x{:016x} = {}@{}'.format(address, self.symbols[base][offset], base))
                            return self.symbols[base][offset]
                        elif address in self.symbols[base]:
                            debug.info('Instruction was a call to 0x{:016x} = {}@{}'.format(address, self.symbols[base][address], base))
                            return self.symbols[base][address]
                    break
        return None

    def find_function_address(self, proc_as, ret_addr):
        if False:
            print('Hello World!')
        '\n        Calculates the function address given a return address. Disassembles code to get through the double indirection\n        introduced by the Linux PLT.\n        @param proc_as: Process address space\n        @param ret_addr: Return address\n        @return The function address or None\n        '
        if distorm_loaded:
            decode_as = self.decode_as
            retaddr_assembly = distorm3.Decode(ret_addr - 5, proc_as.read(ret_addr - 5, 5), decode_as)
            if len(retaddr_assembly) == 0:
                return None
            retaddr_assembly = retaddr_assembly[0]
            instr = retaddr_assembly[2].split(' ')
            if instr[0] == 'CALL':
                try:
                    target = int(instr[1][2:], 16)
                except ValueError:
                    return None
                bytes = proc_as.read(target, 6)
                if not bytes:
                    return target
                plt_instructions = distorm3.Decode(target, bytes, decode_as)
                plt_assembly = plt_instructions[0]
                instr2 = plt_assembly[2].split(' ')
                if instr2[0] == 'JMP':
                    final_addr = None
                    if instr2[1] == 'DWORD':
                        target2 = int(instr2[2][3:-1], 16)
                    elif instr2[1] == 'QWORD':
                        target2 = int(instr2[2][7:-1], 16)
                    else:
                        final_addr = int(instr2[1][2:], 16)
                    if not final_addr:
                        final_addr = target + 6 + target2
                    debug.info('Found function address from instruction {} at offset 0x{:016x}'.format(instr2, target))
                    return read_address(proc_as, final_addr)
                elif instr2[0] == 'PUSH' and instr2[1] == 'RBP':
                    debug.info('Found function address from instruction {} at offset 0x{:016x}'.format(instr, target))
                    return target
                else:
                    debug.info('Found function address from instruction {} at offset 0x{:016x}'.format(instr, target))
                    return target
            return None
        else:
            return None

    def calculate_annotations(self, frames):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create annotations using the frame list.\n        @param frames: a list of stackframes\n        @return a dict of stack address -> (value, annotation)\n        '
        size = linux_process_info.address_size
        end = frames[-1].address
        start = frames[0].ebp_address
        l = linux_process_info.read_int_list(start, end, frames[0].proc_as)
        result = {}
        offset = start
        for value in l:
            result[offset] = (value, '')
            offset += size
        for frame in frames[::-1]:
            result[frame.ebp_address] = (frame.ebp, '')
            annotation = 'return address'
            if frame.function:
                annotation += ' for {:016x}'.format(frame.function)
            if frame.symbol:
                annotation += ' ( {} )'.format(frame.symbol)
            result[frame.ret_address] = (frame.ret, annotation)
        return result

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.outfd = outfd
        for (p, reg, frames) in data:
            debug.info('Found {} frames!'.format(len(frames)))
            debug.info('')
            print(frames)
            if self.dump_file:
                self.write_annotated_stack(self.dump_file, self.calculate_annotations(frames))
        print(stats)

    def write_annotated_stack(self, f, stack_ann):
        if False:
            return 10
        '\n        Writes an annotated to a file ( the -o option )\n        @param f: The file to write\n        @param stack_ann: the annotated stack dict as returned by calculate_annotations()\n        @return: None\n        '
        f.write('{:16s}  {:16s} {}\n'.format('Address', 'Value', 'Annotation'))
        for address in sorted(stack_ann.keys()):
            (value, ann) = stack_ann[address]
            f.write('{:016x}: {:016x} {}\n'.format(address, value, ann))

class stack_frame(object):
    """
    A class to record info about a stack frame.
    """

    def __init__(self, address, proc_as, frame_number):
        if False:
            while True:
                i = 10
        self.address = address
        self.proc_as = proc_as
        self.frame_number = frame_number
        self._function = None
        self.symbol = None
        self.locals_size = None

    @property
    def function(self):
        if False:
            for i in range(10):
                print('nop')
        return self._function

    @function.setter
    def function(self, value):
        if False:
            return 10
        self._function = value

    @property
    def ret(self):
        if False:
            print('Hello World!')
        if self.proc_as.is_valid_address(self.ret_address):
            return read_address(self.proc_as, self.ret_address)
        return 0

    @property
    def ret_address(self):
        if False:
            return 10
        return self.address - linux_process_info.address_size

    @property
    def ebp(self):
        if False:
            print('Hello World!')
        if self.proc_as.is_valid_address(self.ebp_address) and self.ebp_address != 0:
            return read_address(self.proc_as, self.ebp_address)
        return 0

    @property
    def ebp_address(self):
        if False:
            while True:
                i = 10
        return self.address - linux_process_info.address_size * 2

    @property
    def arg_address(self):
        if False:
            for i in range(10):
                print('nop')
        return self.address - linux_process_info.address_size * 3

    @property
    def locals_end(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ret_address - self.locals_size

    def get_locals(self):
        if False:
            return 10
        start = self.locals_end - linux_process_info.address_size
        end = self.ret_address - linux_process_info.address_size
        return linux_process_info.read_int_list(start, end, self.proc_as)

    def __repr__(self):
        if False:
            return 10
        rep = '\n'
        rep += 'Frame {}\n========\n'.format(self.frame_number)
        rep += 'Stack frame at 0x{:016x}\n'.format(self.address)
        if self.locals_size:
            rep += 'Local variables at {:016x} to {:016x}\n'.format(self.ebp_address, self.locals_end)
            if verbose_stack_arguments:
                rep += 'Locals:\n'
                for local in self.get_locals():
                    rep += '\t0x{:016x}\n'.format(local)
        rep += 'Saved registers:\n'
        rep += '\tebp at 0x{:016x}: 0x{:016x}\n'.format(self.ebp_address, self.ebp)
        rep += '\teip at 0x{:016x}: 0x{:016x} (Return Address)\n'.format(self.ret_address, self.ret)
        if self.function:
            rep += 'Frame function address: {:016x}\n'.format(self.function)
        if self.symbol:
            rep += 'Frame function symbol: {}\n'.format(self.symbol)
        return rep