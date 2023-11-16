from __future__ import annotations
import argparse
import pwndbg.color.memory as M
import pwndbg.commands
import pwndbg.gdblib.arch
import pwndbg.gdblib.config
import pwndbg.gdblib.memory
import pwndbg.gdblib.regs
import pwndbg.gdblib.stack
import pwndbg.gdblib.vmmap
import pwndbg.wrappers
from pwndbg.commands import CommandCategory
parser = argparse.ArgumentParser(description='Shows offsets of the specified address from various useful locations.')
parser.add_argument('address', nargs='?', default='$pc', help='Address to inspect')

def print_line(name, addr, first, second, op, width=20) -> None:
    if False:
        return 10
    print(f"{name.rjust(width)} {M.get(addr)} = {(M.get(first) if not isinstance(first, str) else first.ljust(len(hex(addr).rstrip('L'))))} {op} {second:#x}")

def xinfo_stack(page, addr) -> None:
    if False:
        print('Hello World!')
    sp = pwndbg.gdblib.regs.sp
    frame = pwndbg.gdblib.regs[pwndbg.gdblib.regs.frame]
    frame_mapping = pwndbg.gdblib.vmmap.find(frame)
    print_line('Stack Top', addr, page.vaddr, addr - page.vaddr, '+')
    print_line('Stack End', addr, page.end, page.end - addr, '-')
    print_line('Stack Pointer', addr, sp, addr - sp, '+')
    if frame_mapping and page.vaddr == frame_mapping.vaddr:
        print_line('Frame Pointer', addr, frame, frame - addr, '-')
    canary_value = pwndbg.commands.canary.canary_value()[0]
    if canary_value is not None:
        all_canaries = list(pwndbg.search.search(pwndbg.gdblib.arch.pack(canary_value), mappings=pwndbg.gdblib.stack.stacks.values()))
        follow_canaries = sorted(filter(lambda a: a > addr, all_canaries))
        if follow_canaries is not None and len(follow_canaries) > 0:
            nxt = follow_canaries[0]
            print_line('Next Stack Canary', addr, nxt, nxt - addr, '-')

def xinfo_mmap_file(page, addr) -> None:
    if False:
        return 10
    file_name = page.objfile
    objpages = filter(lambda p: p.objfile == file_name, pwndbg.gdblib.vmmap.get())
    first = sorted(objpages, key=lambda p: p.vaddr)[0]
    rva = addr - first.vaddr
    print_line('File (Base)', addr, first.vaddr, rva, '+')
    containing_loads = [seg for seg in pwndbg.gdblib.elf.get_containing_segments(file_name, first.vaddr, addr) if seg['p_type'] == 'PT_LOAD']
    for segment in containing_loads:
        if segment['p_type'] == 'PT_LOAD' and addr < segment['x_vaddr_mem_end']:
            offset = addr - segment['p_vaddr']
            print_line('File (Segment)', addr, segment['p_vaddr'], offset, '+')
            break
    for segment in containing_loads:
        if segment['p_type'] == 'PT_LOAD' and addr < segment['x_vaddr_file_end']:
            file_offset = segment['p_offset'] + (addr - segment['p_vaddr'])
            print_line('File (Disk)', addr, file_name, file_offset, '+')
            break
    else:
        print(f"{'File (Disk)'.rjust(20)} {M.get(addr)} = [not file backed]")
    containing_sections = pwndbg.gdblib.elf.get_containing_sections(file_name, first.vaddr, addr)
    if len(containing_sections) > 0:
        print('\n Containing ELF sections:')
        for sec in containing_sections:
            print_line(sec['x_name'], addr, sec['sh_addr'], addr - sec['sh_addr'], '+')

def xinfo_default(page, addr) -> None:
    if False:
        return 10
    print_line('Mapped Area', addr, page.vaddr, addr - page.vaddr, '+')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.MEMORY)
@pwndbg.commands.OnlyWhenRunning
def xinfo(address=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    address = address.cast(pwndbg.gdblib.typeinfo.pvoid)
    addr = int(address)
    addr &= pwndbg.gdblib.arch.ptrmask
    page = pwndbg.gdblib.vmmap.find(addr)
    if page is None:
        print(f'\n  Virtual address {addr:#x} is not mapped.')
        return
    print(f'Extended information for virtual address {M.get(addr)}:')
    print('\n  Containing mapping:')
    print(M.get(address, text=str(page)))
    print('\n  Offset information:')
    if page.is_stack:
        xinfo_stack(page, addr)
    else:
        xinfo_default(page, addr)
    if page.is_memory_mapped_file:
        xinfo_mmap_file(page, addr)