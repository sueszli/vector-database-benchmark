"""
Routines to enumerate mapped memory, and attempt to associate
address ranges with various ELF files and permissions.

The reason that we need robustness is that not every operating
system has /proc/$$/maps, which backs 'info proc mapping'.
"""
from __future__ import annotations
import bisect
from typing import Any
import gdb
import pwndbg.color.message as M
import pwndbg.gdblib.abi
import pwndbg.gdblib.elf
import pwndbg.gdblib.events
import pwndbg.gdblib.file
import pwndbg.gdblib.info
import pwndbg.gdblib.memory
import pwndbg.gdblib.proc
import pwndbg.gdblib.qemu
import pwndbg.gdblib.regs
import pwndbg.gdblib.remote
import pwndbg.gdblib.stack
import pwndbg.gdblib.typeinfo
import pwndbg.lib.cache
explored_pages: list[pwndbg.lib.memory.Page] = []
custom_pages: list[pwndbg.lib.memory.Page] = []
kernel_vmmap_via_pt = pwndbg.gdblib.config.add_param('kernel-vmmap-via-page-tables', 'deprecated', 'the deprecated config of the method get kernel vmmap', help_docstring='Deprecated in favor of `kernel-vmmap`')
kernel_vmmap = pwndbg.gdblib.config.add_param('kernel-vmmap', 'page-tables', 'the method to get vmmap information when debugging via QEMU kernel', help_docstring="kernel-vmmap can be:\npage-tables    - read /proc/$qemu-pid/mem to parse kernel page tables to render vmmap\nmonitor        - use QEMU's `monitor info mem` to render vmmap\nnone           - disable vmmap rendering; useful if rendering is particularly slow\n\nNote that the page-tables method will require the QEMU kernel process to be on the same machine and within the same PID namespace. Running QEMU kernel and GDB in different Docker containers will not work. Consider running both containers with --pid=host (meaning they will see and so be able to interact with all processes on the machine).\n", param_class=gdb.PARAM_ENUM, enum_sequence=['page-tables', 'monitor', 'none'])

@pwndbg.lib.cache.cache_until('objfile', 'start')
def is_corefile() -> bool:
    if False:
        return 10
    "\n    For example output use:\n        gdb ./tests/binaries/crash_simple.out -ex run -ex 'generate-core-file ./core' -ex 'quit'\n\n    And then use:\n        gdb ./tests/binaries/crash_simple.out -core ./core -ex 'info target'\n    And:\n        gdb -core ./core\n\n    As the two differ in output slighty.\n    "
    return 'Local core dump file:\n' in pwndbg.gdblib.info.target()
inside_no_proc_maps_search = False

@pwndbg.lib.cache.cache_until('start', 'stop')
def get() -> tuple[pwndbg.lib.memory.Page, ...]:
    if False:
        print('Hello World!')
    '\n    Returns a tuple of `Page` objects representing the memory mappings of the\n    target, sorted by virtual address ascending.\n    '
    if not pwndbg.gdblib.proc.alive:
        return tuple()
    if is_corefile():
        return tuple(coredump_maps())
    proc_maps = None
    if pwndbg.gdblib.qemu.is_qemu_usermode():
        proc_maps = info_proc_maps()
    if not proc_maps:
        proc_maps = proc_pid_maps()
    if proc_maps is not None:
        return proc_maps
    pages = []
    if pwndbg.gdblib.qemu.is_qemu_kernel() and pwndbg.gdblib.arch.current in ('i386', 'x86-64', 'aarch64', 'rv32', 'rv64'):
        if kernel_vmmap_via_pt != 'deprecated':
            print(M.warn('`kernel-vmmap-via-page-tables` is deprecated, please use `kernel-vmmap` instead.'))
        if kernel_vmmap == 'page-tables':
            pages.extend(kernel_vmmap_via_page_tables())
        elif kernel_vmmap == 'monitor':
            pages.extend(kernel_vmmap_via_monitor_info_mem())
    global inside_no_proc_maps_search
    if not pages and (not inside_no_proc_maps_search):
        inside_no_proc_maps_search = True
        pages.extend(info_auxv())
        if pages:
            pages.extend(info_sharedlibrary())
        else:
            if pwndbg.gdblib.qemu.is_qemu():
                return (pwndbg.lib.memory.Page(0, pwndbg.gdblib.arch.ptrmask, 7, 0, '[qemu]'),)
            pages.extend(info_files())
        pages.extend(pwndbg.gdblib.stack.stacks.values())
        inside_no_proc_maps_search = False
    pages.extend(explored_pages)
    pages.extend(custom_pages)
    pages.sort()
    return tuple(pages)

@pwndbg.lib.cache.cache_until('stop')
def find(address):
    if False:
        while True:
            i = 10
    if address is None:
        return None
    address = int(address)
    for page in get():
        if address in page:
            return page
    return explore(address)

@pwndbg.gdblib.abi.LinuxOnly()
def explore(address_maybe: int) -> Any | None:
    if False:
        return 10
    '\n    Given a potential address, check to see what permissions it has.\n\n    Returns:\n        Page object\n\n    Note:\n        Adds the Page object to a persistent list of pages which are\n        only reset when the process dies.  This means pages which are\n        added this way will not be removed when unmapped.\n\n        Also assumes the entire contiguous section has the same permission.\n    '
    if proc_pid_maps():
        return None
    address_maybe = pwndbg.lib.memory.page_align(address_maybe)
    flags = 4 if pwndbg.gdblib.memory.peek(address_maybe) else 0
    if not flags:
        return None
    flags |= 2 if pwndbg.gdblib.memory.poke(address_maybe) else 0
    flags |= 1 if not pwndbg.gdblib.stack.nx else 0
    page = find_boundaries(address_maybe)
    page.objfile = '<explored>'
    page.flags = flags
    explored_pages.append(page)
    return page

def explore_registers() -> None:
    if False:
        i = 10
        return i + 15
    for regname in pwndbg.gdblib.regs.common:
        find(pwndbg.gdblib.regs[regname])

def clear_explored_pages() -> None:
    if False:
        return 10
    while explored_pages:
        explored_pages.pop()

def add_custom_page(page) -> None:
    if False:
        return 10
    bisect.insort(custom_pages, page)
    pwndbg.lib.cache.clear_caches()

def clear_custom_page() -> None:
    if False:
        print('Hello World!')
    while custom_pages:
        custom_pages.pop()
    pwndbg.lib.cache.clear_caches()

@pwndbg.lib.cache.cache_until('objfile', 'start')
def coredump_maps():
    if False:
        print('Hello World!')
    '\n    Parses `info proc mappings` and `maintenance info sections`\n    and tries to make sense out of the result :)\n    '
    pages = []
    try:
        info_proc_mappings = pwndbg.gdblib.info.proc_mappings().splitlines()
    except gdb.error:
        info_proc_mappings = []
    for line in info_proc_mappings:
        try:
            (start, _end, size, offset, objfile) = line.split()
            (start, size, offset) = (int(start, 16), int(size, 16), int(offset, 16))
        except (IndexError, ValueError):
            continue
        pages.append(pwndbg.lib.memory.Page(start, size, 0, offset, objfile))
    started_sections = False
    for line in gdb.execute('maintenance info sections', to_string=True).splitlines():
        if not started_sections:
            if 'Core file:' in line:
                started_sections = True
            continue
        try:
            (_idx, start_end, _at_str, _at, name, *flags_list) = line.split()
            (start, end) = map(lambda v: int(v, 16), start_end.split('->'))
            if start == 0:
                continue
            offset = 0
        except (IndexError, ValueError):
            continue
        flags = 0
        if 'READONLY' in flags_list:
            flags |= 4
        if 'DATA' in flags_list:
            flags |= 2
        if 'CODE' in flags_list:
            flags |= 1
        known_page = False
        for page in pages:
            if start in page:
                page.flags |= flags
                known_page = True
                break
        if known_page:
            continue
        pages.append(pwndbg.lib.memory.Page(start, end - start, flags, offset, name))
    if not pages:
        return tuple()
    vsyscall_page = pages[-1]
    if vsyscall_page.start > 18446744073692774400 and vsyscall_page.flags & 1:
        vsyscall_page.objfile = '[vsyscall]'
        vsyscall_page.offset = 0
    stack_addr = None
    auxv = pwndbg.gdblib.info.auxv().splitlines()
    for line in auxv:
        if 'AT_EXECFN' in line:
            try:
                stack_addr = int(line.split()[-2], 16)
            except Exception as e:
                pass
            break
    if stack_addr is not None:
        for page in pages:
            if stack_addr in page:
                page.objfile = '[stack]'
                page.flags |= 6
                page.offset = 0
                break
    return tuple(pages)

@pwndbg.lib.cache.cache_until('start', 'stop')
def info_proc_maps():
    if False:
        i = 10
        return i + 15
    '\n    Parse the result of info proc mappings.\n    Returns:\n        A tuple of pwndbg.lib.memory.Page objects or None if\n        info proc mapping is not supported on the target.\n    '
    try:
        info_proc_mappings = pwndbg.gdblib.info.proc_mappings().splitlines()
    except gdb.error:
        info_proc_mappings = []
    pages = []
    for line in info_proc_mappings:
        try:
            split_line = line.split()
            if len(split_line) < 6:
                (start, _end, size, offset, objfile) = split_line
                perm = 'rwxp'
            else:
                (start, _end, size, offset, perm, objfile) = split_line
            (start, size, offset) = (int(start, 16), int(size, 16), int(offset, 16))
        except (IndexError, ValueError):
            continue
        flags = 0
        if 'r' in perm:
            flags |= 4
        if 'w' in perm:
            flags |= 2
        if 'x' in perm:
            flags |= 1
        pages.append(pwndbg.lib.memory.Page(start, size, flags, offset, objfile))
    return tuple(pages)

@pwndbg.lib.cache.cache_until('start', 'stop')
def proc_pid_maps():
    if False:
        print('Hello World!')
    "\n    Parse the contents of /proc/$PID/maps on the server.\n\n    Returns:\n        A tuple of pwndbg.lib.memory.Page objects or None if\n        /proc/$pid/maps doesn't exist or when we debug a qemu-user target\n    "
    if pwndbg.gdblib.qemu.is_qemu():
        return None
    pid = pwndbg.gdblib.proc.pid
    locations = [f'/proc/{pid}/maps', f'/proc/{pid}/map', f'/usr/compat/linux/proc/{pid}/maps']
    for location in locations:
        try:
            data = pwndbg.gdblib.file.get(location).decode()
            break
        except (OSError, gdb.error):
            continue
    else:
        return None
    if data == '':
        return tuple()
    pages = []
    for line in data.splitlines():
        (maps, perm, offset, dev, inode_objfile) = line.split(maxsplit=4)
        (start, stop) = maps.split('-')
        try:
            (inode, objfile) = inode_objfile.split(maxsplit=1)
        except Exception:
            objfile = '[anon_' + start[:-3] + ']'
        start = int(start, 16)
        stop = int(stop, 16)
        offset = int(offset, 16)
        size = stop - start
        flags = 0
        if 'r' in perm:
            flags |= 4
        if 'w' in perm:
            flags |= 2
        if 'x' in perm:
            flags |= 1
        page = pwndbg.lib.memory.Page(start, size, flags, offset, objfile)
        pages.append(page)
    return tuple(pages)

@pwndbg.lib.cache.cache_until('stop')
def kernel_vmmap_via_page_tables():
    if False:
        while True:
            i = 10
    import pt
    retpages: list[pwndbg.lib.memory.Page] = []
    p = pt.PageTableDump()
    try:
        p.lazy_init()
    except Exception:
        print(M.error('Permission error when attempting to parse page tables with gdb-pt-dump.\n' + 'Either change the kernel-vmmap setting, re-run GDB as root, or disable `ptrace_scope` (`echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`)'))
        return tuple(retpages)
    if not pwndbg.gdblib.kernel.paging_enabled():
        return tuple(retpages)
    pages = p.backend.parse_tables(p.cache, p.parser.parse_args(''))
    for page in pages:
        start = page.va
        size = page.page_size
        flags = 4
        if page.pwndbg_is_writeable():
            flags |= 2
        if page.pwndbg_is_executable():
            flags |= 1
        objfile = f'[pt_{hex(start)[2:-3]}]'
        retpages.append(pwndbg.lib.memory.Page(start, size, flags, 0, objfile))
    return tuple(retpages)
monitor_info_mem_not_warned = True

def kernel_vmmap_via_monitor_info_mem():
    if False:
        return 10
    '\n    Returns Linux memory maps information by parsing `monitor info mem` output\n    from QEMU kernel GDB stub.\n    Works only on X86/X64/RISC-V as this is what QEMU supports.\n\n    Consider using the `kernel_vmmap_via_page_tables` method\n    as it is probably more reliable/better.\n\n    See also: https://github.com/pwndbg/pwndbg/pull/685\n    (TODO: revisit with future QEMU versions)\n\n    # Example output from the command:\n    # pwndbg> monitor info mem\n    # ffff903580000000-ffff903580099000 0000000000099000 -rw\n    # ffff903580099000-ffff90358009b000 0000000000002000 -r-\n    # ffff90358009b000-ffff903582200000 0000000002165000 -rw\n    # ffff903582200000-ffff903582803000 0000000000603000 -r-\n    '
    global monitor_info_mem_not_warned
    monitor_info_mem = None
    try:
        monitor_info_mem = gdb.execute('monitor info mem', to_string=True)
    finally:
        if monitor_info_mem is None or 'unknown command' in monitor_info_mem:
            if pwndbg.gdblib.arch.name == 'aarch64':
                print(M.error(f'The {pwndbg.gdblib.arch.name} architecture does' + ' not support the `monitor info mem` command. Run ' + '`help show kernel-vmmap` for other options.'))
            return tuple()
    lines = monitor_info_mem.splitlines()
    if len(lines) == 1 and lines[0] == 'PG disabled':
        return tuple()
    pages = []
    for line in lines:
        dash_idx = line.index('-')
        space_idx = line.index(' ')
        rspace_idx = line.rindex(' ')
        start = int(line[:dash_idx], 16)
        end = int(line[dash_idx + 1:space_idx], 16)
        size = int(line[space_idx + 1:rspace_idx], 16)
        if end - start != size and monitor_info_mem_not_warned:
            print(M.warn('The vmmap output may be incorrect as `monitor info mem` output assertion/assumption\nthat end-start==size failed. The values are:\nend=%#x; start=%#x; size=%#x; end-start=%#x\nNote that this warning will not show up again in this Pwndbg/GDB session.' % (end, start, size, end - start)))
            monitor_info_mem_not_warned = False
        perm = line[rspace_idx + 1:]
        flags = 0
        if 'r' in perm:
            flags |= 4
        if 'w' in perm:
            flags |= 2
        flags |= 1
        pages.append(pwndbg.lib.memory.Page(start, size, flags, 0, '<qemu>'))
    return tuple(pages)

@pwndbg.lib.cache.cache_until('stop')
def info_sharedlibrary():
    if False:
        print('Hello World!')
    '\n    Parses the output of `info sharedlibrary`.\n\n    Specifically, all we really want is any valid pointer into each library,\n    and the path to the library on disk.\n\n    With this information, we can use the ELF parser to get all of the\n    page permissions for every mapped page in the ELF.\n\n    Returns:\n        A list of pwndbg.lib.memory.Page objects.\n    '
    pages = []
    for line in pwndbg.gdblib.info.sharedlibrary().splitlines():
        if not line.startswith('0x'):
            continue
        tokens = line.split()
        text = int(tokens[0], 16)
        obj = tokens[-1]
        pages.extend(pwndbg.gdblib.elf.map(text, obj))
    return tuple(sorted(pages))

@pwndbg.lib.cache.cache_until('stop')
def info_files():
    if False:
        print('Hello World!')
    seen_files = set()
    pages = []
    main_exe = ''
    for line in pwndbg.gdblib.info.files().splitlines():
        line = line.strip()
        if line.startswith('`'):
            (exename, filetype) = line.split(maxsplit=1)
            main_exe = exename.strip("`,'")
            continue
        if not line.startswith('0x'):
            continue
        fields = line.split(maxsplit=6)
        vaddr = int(fields[0], 16)
        if len(fields) == 5:
            objfile = main_exe
        elif len(fields) == 7:
            objfile = fields[6]
        else:
            print('Bad data: %r' % line)
            continue
        if objfile in seen_files:
            continue
        else:
            seen_files.add(objfile)
        pages.extend(pwndbg.gdblib.elf.map(vaddr, objfile))
    return tuple(pages)

@pwndbg.lib.cache.cache_until('exit')
def info_auxv(skip_exe: bool=False):
    if False:
        return 10
    '\n    Extracts the name of the executable from the output of the command\n    "info auxv". Note that if the executable path is a symlink,\n    it is not dereferenced by `info auxv` and we also don\'t dereference it.\n\n    Arguments:\n        skip_exe(bool): Do not return any mappings that belong to the exe.\n\n    Returns:\n        A list of pwndbg.lib.memory.Page objects.\n    '
    auxv = pwndbg.auxv.get()
    if not auxv:
        return tuple()
    pages = []
    exe_name = auxv.AT_EXECFN or 'main.exe'
    entry = auxv.AT_ENTRY
    base = auxv.AT_BASE
    vdso = auxv.AT_SYSINFO_EHDR or auxv.AT_SYSINFO
    phdr = auxv.AT_PHDR
    if not skip_exe and (entry or phdr):
        for addr in [entry, phdr]:
            if not addr:
                continue
            new_pages = pwndbg.gdblib.elf.map(addr, exe_name)
            if new_pages:
                pages.extend(new_pages)
                break
    if base:
        pages.extend(pwndbg.gdblib.elf.map(base, '[linker]'))
    if vdso:
        pages.extend(pwndbg.gdblib.elf.map(vdso, '[vdso]'))
    return tuple(sorted(pages))

def find_boundaries(addr, name: str='', min: int=0):
    if False:
        return 10
    '\n    Given a single address, find all contiguous pages\n    which are mapped.\n    '
    start = pwndbg.gdblib.memory.find_lower_boundary(addr)
    end = pwndbg.gdblib.memory.find_upper_boundary(addr)
    start = max(start, min)
    return pwndbg.lib.memory.Page(start, end - start, 4, 0, name)

def check_aslr():
    if False:
        i = 10
        return i + 15
    "\n    Detects the ASLR status. Returns True, False or None.\n\n    None is returned when we can't detect ASLR.\n    "
    if pwndbg.gdblib.qemu.is_qemu():
        return (None, 'Could not detect ASLR on QEMU targets')
    try:
        data = pwndbg.gdblib.file.get('/proc/sys/kernel/randomize_va_space')
        if b'0' in data:
            return (False, 'kernel.randomize_va_space == 0')
    except Exception as e:
        print("Could not check ASLR: can't read randomize_va_space")
    if pwndbg.gdblib.proc.alive:
        try:
            data = pwndbg.gdblib.file.get('/proc/%i/personality' % pwndbg.gdblib.proc.pid)
            personality = int(data, 16)
            return (personality & 262144 == 0, "read status from process' personality")
        except Exception:
            print("Could not check ASLR: can't read process' personality")
    output = gdb.execute('show disable-randomization', to_string=True)
    return ('is off.' in output, 'show disable-randomization')