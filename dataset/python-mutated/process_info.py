"""
@author:       Edwin Smulders
@license:      GNU General Public License 2.0 or later
@contact:      mail@edwinsmulders.eu
"""
import struct
import collections
import itertools
import volatility.plugins.linux.pslist as linux_pslist
import volatility.plugins.linux.proc_maps as linux_proc_maps
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.threads as linux_threads
registers = collections.namedtuple('registers', ['r15', 'r14', 'r13', 'r12', 'rbp', 'rbx', 'r11', 'r10', 'r9', 'r8', 'rax', 'rcx', 'rdx', 'rsi', 'rdi', 'unknown', 'rip', 'cs', 'eflags', 'rsp', 'ss'])
address_size = 8

def null_list(pages, size):
    if False:
        return 10
    '\n    Split a section (divided by pages) on 0-bytes.\n\n    @param pages: a list of pages\n    @param size: total size of the section\n    @return: a list of strings\n    '
    res = []
    for page in pages:
        if size > 4096:
            size -= 4096
        else:
            page = page[:size]
            for s in page.split('\x00'):
                if s != '':
                    res.append(s)
    return res

def int_list(pages, size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split a range into integers. Will split into words (e.g. 4 or 8 bytes).\n\n    @param pages: a list of pages\n    @param size: total size of the section\n    @return: a list of word-sized integers\n    '
    if address_size == 4:
        fmt = '<L'
    else:
        fmt = '<Q'
    for page in pages:
        curr = 0
        while curr < 4096 and curr < size:
            yield struct.unpack(fmt, page[curr:curr + address_size])[0]
            curr += address_size

def _neg_fix(addr):
    if False:
        i = 10
        return i + 15
    return addr

def print_hex(value):
    if False:
        return 10
    'Print a value as in 4 byte hexadecimal.'
    print('0x{:08x}'.format(value))

def read_addr_range(start, end, addr_space):
    if False:
        i = 10
        return i + 15
    '\n    Read a number of pages.\n\n    @param start: Start address\n    @param end: End address\n    @param addr_space: The virtual address space\n    @return: a list of pages\n    '
    pagesize = 4096
    while start < end:
        page = addr_space.zread(start, pagesize)
        yield page
        start += pagesize

def read_null_list(start, end, addr_space):
    if False:
        print('Hello World!')
    '\n    Read a number of pages and split it on 0-bytes.\n\n    @param start: Start address\n    @param end: End address\n    @param addr_space: The virtual address space\n    @return: a list of strings\n    '
    return null_list(read_addr_range(start, end, addr_space), end - start)

def read_int_list(start, end, addr_space):
    if False:
        while True:
            i = 10
    '\n    Read a number of pages and split it into integers.\n\n    @param start: Start address\n    @param end: End address\n    @param addr_space: The virtual address space\n    @return: a list of integers.\n    '
    return int_list(read_addr_range(start, end, addr_space), end - start)

def read_registers(task, addr_space):
    if False:
        i = 10
        return i + 15
    '\n    Read registers from kernel space. Needs to be replaced by the linux_info_regs plugin.\n\n    @param task: The relevant task_struct\n    @param addr_space: The kernel address space\n    @return: A list of registers (integers)\n    '
    return list(read_int_list(task.thread.sp0 - 21 * address_size, task.thread.sp0, addr_space))

class linux_process_info:
    """ Plugin to gather info for a task/process. Extends pslist. """

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        linux_common.set_plugin_members(self)
        global address_size
        if self.profile.metadata.get('memory_model', '32bit') == '32bit':
            address_size = 4
        else:
            address_size = 8
        self.get_threads = linux_threads.linux_threads(config).get_threads

    def read_addr_range(self, start, end, addr_space=None):
        if False:
            return 10
        ' Read an address range with the task address space as default.\n\n        @param start: Start address\n        @param end: End address\n        @param addr_space: The address space to read.\n        @return: a list of pages\n        '
        if addr_space == None:
            addr_space = self.proc_as
        return read_addr_range(start, end, addr_space)

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            self.task = task
            yield self.analyze(task)

    def read_null_list(self, start, end, addr_space=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read a number of pages and split it on 0-bytes, with the task address space as default.\n\n        @param start: Start address\n        @param end: End address\n        @param addr_space: The virtual address space\n        @return: a list of strings\n        '
        return null_list(self.read_addr_range(start, end, addr_space), end - start)

    def read_int_list(self, start, end, addr_space=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read a number of pages and split it into integers, with the task addres space as default.\n\n        @param start: Start address\n        @param end: End address\n        @param addr_space: The virtual address space\n        @return: a list of integers.\n        '
        return int_list(self.read_addr_range(start, end, addr_space), end - start)

    def analyze(self, task):
        if False:
            while True:
                i = 10
        '\n        Analyze a task_struct.\n\n        @param task: the task_struct\n        @return: a process_info object\n        '
        self.proc_as = task.get_process_address_space()
        p = process_info(task)
        p.kernel_as = self.addr_space
        p.maps = list(task.get_proc_maps())
        if len(p.maps) == 0:
            return None
        for m in p.maps:
            if m.vm_start <= task.mm.start_stack <= m.vm_end:
                p.vm_stack_low = m.vm_start
                p.vm_stack_high = m.vm_end
        if not p.vm_stack_low:
            last = p.maps[-1]
            p.vm_stack_high = last.vm_end
            p.vm_stack_low = last.vm_start
        p.env = self.read_null_list(_neg_fix(task.mm.env_start), _neg_fix(task.mm.env_end))
        p.stack = self.read_int_list(_neg_fix(p.vm_stack_low), _neg_fix(task.mm.start_stack))
        p.rest_stack = self.read_int_list(_neg_fix(task.mm.start_stack), _neg_fix(task.mm.env_start))
        p.args = self.read_null_list(_neg_fix(task.mm.arg_start), _neg_fix(task.mm.arg_end))
        reglist = read_registers(task, self.addr_space)
        p.reg = registers(*reglist)
        p.threads = self.get_threads(task)[1]
        return p

    def get_map(self, task, address):
        if False:
            while True:
                i = 10
        '\n        Get the vm_area to which an address points.\n\n        @param task: the task_struct\n        @param address: an address\n        @return: a vm_area_struct corresponding to the address\n        '
        for m in task.get_proc_maps():
            if m.vm_start <= address <= m.vm_end:
                return m

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.outfd = outfd

    def render_stack_frames(self, stack_frames):
        if False:
            return 10
        '\n        Render stackframes (old code)\n        @param stack_frames: a list of stackframes\n        @return: None\n        '
        for stack_frame in stack_frames:
            self.table_header(self.outfd, [('Stack Frame', '16'), ('Value', '[addrpad]')])
            self.table_row(self.outfd, 'Frame Number', stack_frame.frame_number)
            self.table_row(self.outfd, 'Offset', stack_frame.offset)
            self.table_row(self.outfd, 'Return Address', stack_frame.ret)

    def render_registers(self, reg):
        if False:
            print('Hello World!')
        '\n        Render a registers named tuple.\n        @param reg: registers named tuple\n        @return: None\n        '
        self.table_header(self.outfd, [('Register', '8'), ('Value', '[addrpad]')])
        for k in reg._fields:
            self.table_row(self.outfd, k, getattr(reg, k))

    def render_list(self, l):
        if False:
            while True:
                i = 10
        '\n        Render an address list\n        @param l: address list\n        @return: None\n        '
        self.table_header(self.outfd, [('Address', '[addrpad]'), ('Value', '[addrpad]')])
        for (address, value) in l:
            self.table_row(self.outfd, address, value)

    def render_annotated_list(self, ann_list):
        if False:
            print('Hello World!')
        '\n        Render a list including annotations.\n        @param ann_list: a 3-tuple list\n        @return: None\n        '
        self.table_header(self.outfd, [('Address', '[addrpad]'), ('Value', '[addrpad]'), ('Annotation', '50')])
        for (address, value, annotation) in ann_list:
            self.table_row(self.outfd, address, value, annotation)

class process_info(object):
    """
    A class to collect various information about a process/task. Includes helper functions to detect pointers.
    """

    def __init__(self, task):
        if False:
            for i in range(10):
                print('nop')
        self.task = task
        self.mm = task.mm
        self.mm_brk = _neg_fix(self.mm.brk)
        self.mm_end_code = _neg_fix(self.mm.end_code)
        self.mm_end_data = _neg_fix(self.mm.end_data)
        self.mm_env_end = _neg_fix(self.mm.env_end)
        self.mm_start_brk = _neg_fix(self.mm.start_brk)
        self.mm_start_code = _neg_fix(self.mm.start_code)
        self.mm_start_data = _neg_fix(self.mm.start_data)
        self.proc_as = task.get_process_address_space()
        self.kernel_as = None
        self.env = None
        self.rest_stack = None
        self.args = None
        self.vm_stack_low = None
        self.vm_stack_high = None
        self.stack_frames = None
        self.thread_stacks = None
        self.thread_stack_ranges = None
        self._stack = None
        self._threads = None
        self._reg = None
        self._real_stack_low = None
        self._maps = None
        self._exec_maps = None
        self._exec_maps_ranges = None
        self.is_pointer_dict = dict(stack=self.is_stack_pointer, heap=self.is_heap_pointer, constant=self.is_constant_pointer, code=self.is_code_pointer)

    @property
    def maps(self):
        if False:
            i = 10
            return i + 15
        '\n        @return: the vm_area maps list.\n        '
        return self._maps

    @maps.setter
    def maps(self, value):
        if False:
            while True:
                i = 10
        '\n        Setter for maps. Also initializes some other values.\n        @param value: The list of vm_area maps\n        @return: None\n        '
        self._maps = value
        self._exec_maps = []
        self._exec_maps_ranges = []
        for m in self._maps:
            if m.vm_flags.is_executable():
                self._exec_maps.append(m)
                self._exec_maps_ranges.append((m.vm_start, m.vm_end))

    @property
    def reg(self):
        if False:
            return 10
        '\n        @return: the registers named tuple for this process\n        '
        return self._reg

    @reg.setter
    def reg(self, value):
        if False:
            print('Hello World!')
        '\n        Setter for reg.\n        @param value: The named tuple for registers.\n        @return: None\n        '
        self._reg = value

    @property
    def stack(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the _list_ of stack values (old code).\n        @return: stack integer list.\n        '
        return self._stack

    @stack.setter
    def stack(self, value):
        if False:
            print('Hello World!')
        '\n        Set the stack list (old code).\n        @param value: a list of integers.\n        @return: None\n        '
        self._stack = list(value)
        self._calculate_stack_offset()

    @property
    def threads(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the list of threads for this process.\n        @return: a list of task_structs (threads).\n        '
        return self._threads

    @threads.setter
    def threads(self, value):
        if False:
            return 10
        '\n        Set the list of threads. Initializes the list of register tuples for these threads.\n        @param value: The list of task_structs.\n        @return: None\n        '
        self._threads = value
        self.thread_registers = self._find_thread_registers()
        self._generate_thread_stack_list()

    def _find_thread_registers(self):
        if False:
            return 10
        '\n        Reads the registers from the kernel stack for all threads.\n        @return: list of tuple of registers.\n        '
        reglist = []
        for task in self.threads:
            reglist.append(registers(*read_registers(task, self.kernel_as)))
        return reglist

    def get_stack_value(self, address):
        if False:
            print('Hello World!')
        '\n        Read a value from the stack, by using the stack list (old code).\n        @param address: The address to read.\n        @return: The word at this address.\n        '
        return self.stack[self.get_stack_index(address)]

    def get_stack_index(self, address):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the index on the stack list given an address.\n        @param address: The address to find\n        @return: an index into process_info.stack\n        '
        return (address - self.vm_stack_low) / address_size

    def _generate_thread_stack_list(self):
        if False:
            return 10
        '\n        Makes a list of the stack vm areas for all threads. Uses the register contents.\n        @return: None\n        '
        if not self.threads or not self.maps:
            self.thread_stacks = None
        else:
            thread_sps = [self.thread_registers[i].rsp for (i, task) in enumerate(self.threads)]
            thread_sps.sort()
            self.thread_stacks = []
            self.thread_stack_ranges = []
            i = 0
            for m in self.maps:
                if i < len(thread_sps) and m.vm_start <= thread_sps[i] <= m.vm_end:
                    self.thread_stacks.append(m)
                    self.thread_stack_ranges.append((m.vm_start, m.vm_end))
                    i += 1

    def _calculate_stack_offset(self):
        if False:
            return 10
        '\n        Calculates the absolute bottom of the stack (everything below is 0). (old code)\n        @return: The lowest stack address.\n        '
        offset = self.vm_stack_low
        for i in self._stack:
            if i != 0:
                self._real_stack_low = offset
                break
            offset += 4
        return self._real_stack_low

    def annotate_addr_list(self, l, offset=None, skip_zero=True):
        if False:
            while True:
                i = 10
        '\n        Annotates a list of addresses with some basic pointer and register information (old code).\n        @param l: list of addresses.\n        @param offset: Offset of the list\n        @param skip_zero:\n        @return: An annotated list\n        '
        if offset == None:
            offset = self.vm_stack_low
        for value in l:
            if value != 0:
                skip_zero = False
            pointer_type = self.get_pointer_type(value)
            annotation = ''
            if pointer_type != None:
                annotation = pointer_type + ' pointer'
            if offset == self.reg.esp:
                annotation += ' && register esp'
            elif offset == self.reg.ebp:
                annotation += ' && register ebp'
            if not skip_zero:
                yield (offset, value, annotation)
            offset += 4

    def is_stack_pointer(self, addr):
        if False:
            i = 10
            return i + 15
        '\n        Check if addr is a pointer to the (main) stack.\n        @param addr: An address\n        @return: True or False\n        '
        return self.vm_stack_low <= addr <= self.mm_env_end

    def is_thread_stack_pointer(self, addr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if addr is a pointer to a thread stack.\n        FIXME: enable checking a specific stack.\n        @param addr: An address\n        @return: True or False\n        '
        for (m_start, m_end) in self.thread_stack_ranges:
            if m_start <= addr <= m_end:
                return True
        return False

    def is_heap_pointer(self, addr):
        if False:
            return 10
        '\n        Check if addr is a pointer to the heap.\n        @param addr: An address\n        @return: True or False\n        '
        return self.mm_start_brk <= addr <= self.mm_brk

    def is_constant_pointer(self, addr):
        if False:
            while True:
                i = 10
        '\n        Check if addr is a pointer to a program constant\n        @param addr: An address\n        @return: True of False\n        '
        return self.mm_start_data <= addr <= self.mm_end_data

    def is_program_code_pointer(self, addr):
        if False:
            i = 10
            return i + 15
        '\n        Check if addr is a pointer to the program code\n        @param addr: An address\n        @return: True of False\n        '
        return self.mm_start_code <= addr <= self.mm_end_code

    def is_library_code_pointer(self, addr):
        if False:
            while True:
                i = 10
        '\n        Check if addr is a pointer to library code\n        @param addr: An address\n        @return: True or False\n        '
        return self.is_code_pointer(addr) and (not self.is_program_code_pointer(addr))

    def is_code_pointer(self, addr):
        if False:
            return 10
        '\n        Check if addr is a pointer to an executable section of memory\n        @param addr: An address\n        @return: True or False\n        '
        for (m_start, m_end) in self._exec_maps_ranges:
            if m_start <= addr <= m_end:
                return True
        return False

    def is_data_pointer(self, addr):
        if False:
            while True:
                i = 10
        '\n        Check if addr points to data (not code)\n        @param addr: An address\n        @return: True or False\n        '
        return self.is_heap_pointer(addr) or self.is_stack_pointer(addr) or self.is_constant_pointer(addr) or self.is_thread_stack_pointer(addr)

    def is_pointer(self, addr, space=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if addr is any sort of pointer\n        @param addr: An address\n        @param space: A choice of stack, heap, etc\n        @return: True or False\n        '
        if not space:
            for func in self.is_pointer_dict.itervalues():
                if func(addr):
                    return True
            return False
        else:
            return self.is_pointer_dict[space]

    def get_map_by_name(self, name, permissions='r-x'):
        if False:
            return 10
        "\n        Find a memory mapping (vm_area) by its name (not exact match). Optionally, check permissions.\n        @param name: The mapped name to find.\n        @param permissions: Permissions in 'rwx' format\n        @return: A (vm_start, vm_end, libname) tuple or None\n        "
        for vma in self.task.get_proc_maps():
            libname = linux_common.get_path(self.task, vma.vm_file)
            if str(vma.vm_flags) == permissions and name in libname:
                return (vma.vm_start, vma.vm_end, libname)
        return None

    def get_unique_data_pointers(self):
        if False:
            print('Hello World!')
        '\n        A filter over get_data_pointers() to get only unique values.\n        @return: A iterator of pointers.\n        '
        return self.get_unique_pointers(self.get_data_pointers())

    def get_unique_pointers(self, pointer_iter=None):
        if False:
            return 10
        '\n        Filter an iterator to only return unique values.\n        @param pointer_iter: The pointer iterator to use. If None, use get_pointers().\n        @return: An iterator of unique pointers\n        '
        if pointer_iter == None:
            pointer_iter = self.get_pointers()
        store = []
        for (address, value) in pointer_iter:
            if value not in store:
                yield (address, value)
                store.append(value)

    def get_data_pointers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls get_pointers with self.is_data_pointer as a filter.\n        @return: An iterator of pointers\n        '
        return self.get_pointers(self.is_data_pointer)

    def get_pointers(self, cond=None, space=None):
        if False:
            print('Hello World!')
        '\n        Finds pointers given a condition and a space. (old code)\n        @param cond: The type of pointer to filter, defaults to self.is_pointer\n        @param space: The list of values to use, defaults to self.stack\n        @return: An iterator of addresses and their values.\n        '
        if cond == None:
            cond = self.is_pointer
        if space == None:
            space = self.stack
        address = self.vm_stack_low
        for value in space:
            if value != 0 and cond(value):
                yield (address, value)
            address += address_size

    def get_data_pointers_from_heap(self):
        if False:
            print('Hello World!')
        '\n        Find data pointers on the heap, very slow.\n        @return: An iterator of pointers\n        '
        return self.get_pointers(cond=self.is_data_pointer, space=read_int_list(self.mm_start_brk, self.mm_brk, self.proc_as))

    def get_data_pointers_from_map(self, m):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find data pointers from a specific mapping, very slow.\n        @param m: The vm_area map\n        @return: An iterator of pointers\n        '
        return self.get_pointers(cond=self.is_data_pointer, space=read_int_list(m.vm_start, m.vm_end, self.proc_as))

    def get_data_pointers_from_threads(self):
        if False:
            i = 10
            return i + 15
        '\n        Find data pointers from all threads\n        @return: An iterator of all pointers on thread stacks\n        '
        iterators = [self.get_data_pointers_from_map(m) for m in self.thread_stacks]
        return self.get_unique_pointers(itertools.chain(*iterators))

    def get_pointers_from_stack(self):
        if False:
            while True:
                i = 10
        '\n        Find pointers on the main stack\n        @return: An iterator of pointers\n        '
        return self.get_pointers(space=self.stack)

    def get_pointer_type(self, addr):
        if False:
            return 10
        '\n        Determine the pointer type for a specific address.\n        @param addr: An address.\n        @return: String pointer type\n        '
        for (k, v) in self.is_pointer_dict.iteritems():
            if v(addr):
                return k
        return None

    def annotated_stack(self):
        if False:
            return 10
        '\n        Uses annotate_addr_list() to annotate the stack.\n        @return: An annotated address list of the stack\n        '
        return self.annotate_addr_list(self._stack)