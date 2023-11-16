from __future__ import annotations
import argparse
import pwndbg.color
import pwndbg.commands
import pwndbg.commands.telescope
import pwndbg.gdblib.arch
import pwndbg.gdblib.memory
from pwndbg.commands import CommandCategory
ts = pwndbg.commands.telescope.telescope

class AddrRange:

    def __init__(self, begin, end) -> None:
        if False:
            while True:
                i = 10
        self.begin = begin
        self.end = end

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return (self.begin, self.end).__repr__()

def get_addrrange_any_named():
    if False:
        for i in range(10):
            print('nop')
    return [AddrRange(page.start, page.end) for page in pwndbg.gdblib.vmmap.get()]

def guess_numbers_base(num: str):
    if False:
        for i in range(10):
            print('nop')
    base = 10
    if num.startswith('0x'):
        base = 16
    elif num.startswith('0b'):
        base = 2
    elif num.startswith('0'):
        base = 8
    return base

def address_range_explicit(section):
    if False:
        i = 10
        return i + 15
    try:
        (begin, end) = section.split(':')
        begin = int(begin, guess_numbers_base(begin))
        end = int(end, guess_numbers_base(end))
        return AddrRange(begin, end)
    except Exception:
        parser.error('"%s" - Bad format of explicit address range! Expected format: "BEGIN_ADDRESS:END_ADDRESS"' % pwndbg.color.red(section))

def address_range(section):
    if False:
        print('Hello World!')
    if section in ('*', 'any'):
        return (0, pwndbg.gdblib.arch.ptrmask)
    if ':' in section:
        return [address_range_explicit(section)]
    pages = list(filter(lambda page: section in page.objfile, pwndbg.gdblib.vmmap.get()))
    if pages:
        return [AddrRange(page.start, page.end) for page in pages]
    else:
        parser.error(f'Memory page with name "{pwndbg.color.red(section)}" does not exist!')
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='Pointer to pointer chain search. Searches given mapping for all pointers that point to specified mapping.\n\nAny chain length greater than 0 is valid. If only one mapping is given it just looks for any pointers in that mapping.')
parser.add_argument('mapping_names', type=address_range, nargs='+', help='Mapping name ')

def maybe_points_to_ranges(ptr: int, rs: list[AddrRange]):
    if False:
        i = 10
        return i + 15
    try:
        pointee = pwndbg.gdblib.memory.pvoid(ptr)
    except Exception:
        return None
    for r in rs:
        if r.begin <= pointee < r.end:
            return pointee
    return None

def p2p_walk(addr, ranges, current_level):
    if False:
        while True:
            i = 10
    levels = len(ranges)
    if current_level >= levels:
        return None
    maybe_addr = maybe_points_to_ranges(addr, ranges[current_level])
    if maybe_addr is None:
        return None
    if current_level == levels - 1:
        return addr
    return p2p_walk(maybe_addr, ranges, current_level + 1)

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.MEMORY)
@pwndbg.commands.OnlyWhenRunning
def p2p(mapping_names: list | None=None) -> None:
    if False:
        i = 10
        return i + 15
    if not mapping_names:
        return
    if len(mapping_names) == 1:
        mapping_names.append(get_addrrange_any_named())
    for rng in mapping_names[0]:
        for addr in range(rng.begin, rng.end):
            maybe_pointer = p2p_walk(addr, mapping_names, current_level=1)
            if maybe_pointer is not None:
                ts(address=addr, count=1)