"""
Compatibility functionality for Windbg users.
"""
from __future__ import annotations
import argparse
import codecs
from itertools import chain
import gdb
import pwndbg.commands
import pwndbg.gdblib.arch
import pwndbg.gdblib.memory
import pwndbg.gdblib.strings
import pwndbg.gdblib.symbol
import pwndbg.gdblib.typeinfo
import pwndbg.hexdump
from pwndbg.commands import CommandCategory
parser = argparse.ArgumentParser(description='Starting at the specified address, dump N bytes.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')
parser.add_argument('count', type=pwndbg.commands.AddressExpr, default=64, nargs='?', help='The number of bytes to dump.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def db(address, count=64):
    if False:
        for i in range(10):
            print('nop')
    '\n    Starting at the specified address, dump N bytes\n    (default 64).\n    '
    return dX(1, address, count, repeat=db.repeat)
parser = argparse.ArgumentParser(description='Starting at the specified address, dump N words.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')
parser.add_argument('count', type=pwndbg.commands.AddressExpr, default=32, nargs='?', help='The number of words to dump.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def dw(address, count=32):
    if False:
        return 10
    '\n    Starting at the specified address, dump N words\n    (default 32).\n    '
    return dX(2, address, count, repeat=dw.repeat)
parser = argparse.ArgumentParser(description='Starting at the specified address, dump N dwords.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')
parser.add_argument('count', type=pwndbg.commands.AddressExpr, default=16, nargs='?', help='The number of dwords to dump.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def dd(address, count=16):
    if False:
        for i in range(10):
            print('nop')
    '\n    Starting at the specified address, dump N dwords\n    (default 16).\n    '
    return dX(4, address, count, repeat=dd.repeat)
parser = argparse.ArgumentParser(description='Starting at the specified address, dump N qwords.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')
parser.add_argument('count', type=pwndbg.commands.AddressExpr, default=8, nargs='?', help='The number of qwords to dump.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def dq(address, count=8):
    if False:
        while True:
            i = 10
    '\n    Starting at the specified address, dump N qwords\n    (default 8).\n    '
    return dX(8, address, count, repeat=dq.repeat)
parser = argparse.ArgumentParser(description='Starting at the specified address, hexdump.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')
parser.add_argument('count', type=pwndbg.commands.AddressExpr, default=8, nargs='?', help='The number of bytes to hexdump.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def dc(address, count=8):
    if False:
        while True:
            i = 10
    return pwndbg.commands.hexdump.hexdump(address=address, count=count)

def dX(size, address, count, to_string=False, repeat=False):
    if False:
        print('Hello World!')
    '\n    Traditionally, windbg will display 16 bytes of data per line.\n    '
    lines = list(chain.from_iterable(pwndbg.hexdump.hexdump(data=None, size=size, count=count, address=address, repeat=repeat, dX_call=True)))
    if not to_string and lines:
        print('\n'.join(lines))
    return lines

def enhex(size, value):
    if False:
        while True:
            i = 10
    value = value & (1 << 8 * size) - 1
    x = '%x' % abs(value)
    x = x.rjust(size * 2, '0')
    return x
parser = argparse.ArgumentParser(description='Write hex bytes at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, nargs='*', help='The bytes to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def eb(address, data):
    if False:
        print('Hello World!')
    '\n    Write hex bytes at the specified address.\n    '
    return eX(1, address, data)
parser = argparse.ArgumentParser(description='Write hex words at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, nargs='*', help='The words to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def ew(address, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write hex words at the specified address.\n    '
    return eX(2, address, data)
parser = argparse.ArgumentParser(description='Write hex dwords at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, nargs='*', help='The dwords to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def ed(address, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write hex dwords at the specified address.\n    '
    return eX(4, address, data)
parser = argparse.ArgumentParser(description='Write hex qwords at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, nargs='*', help='The qwords to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def eq(address, data):
    if False:
        print('Hello World!')
    '\n    Write hex qwords at the specified address.\n    '
    return eX(8, address, data)
parser = argparse.ArgumentParser(description='Write a string at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, help='The string to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def ez(address, data):
    if False:
        i = 10
        return i + 15
    '\n    Write a character at the specified address.\n    '
    return eX(1, address, data, hex=False)
parser = argparse.ArgumentParser(description='Write a string at the specified address.')
parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='The address to write to.')
parser.add_argument('data', type=str, help='The string to write.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def eza(address, data):
    if False:
        print('Hello World!')
    '\n    Write a string at the specified address.\n    '
    return ez(address, data)

def eX(size, address, data, hex=True) -> None:
    if False:
        return 10
    "\n    This relies on windbg's default hex encoding being enforced\n    "
    if not data:
        print('Cannot write empty data into memory.')
        return
    if hex:
        for string in data:
            if string.startswith('0x'):
                string = string[2:]
            if any((ch not in '0123456789abcdefABCDEF' for ch in string)):
                print('Incorrect data format: it must all be a hex value (0x1234 or 1234, both interpreted as 0x1234)')
                return
    writes = 0
    for (i, string) in enumerate(data):
        if hex:
            if string.startswith('0x'):
                string = string[2:]
            string = string.rjust(size * 2, '0')
            data = codecs.decode(string, 'hex')
        else:
            data = string
        if pwndbg.gdblib.arch.endian == 'little':
            data = data[::-1]
        try:
            pwndbg.gdblib.memory.write(address + i * size, data)
            writes += 1
        except gdb.error:
            print('Cannot access memory at address %#x' % address)
            if writes > 0:
                print('(Made %d writes to memory; skipping further writes)' % writes)
            return
parser = argparse.ArgumentParser(description='Dump pointers and symbols at the specified address.')
parser.add_argument('addr', type=pwndbg.commands.HexOrAddressExpr, help='The address to dump from.')

@pwndbg.commands.ArgparsedCommand(parser, aliases=['kd', 'dps', 'dqs'], category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def dds(addr):
    if False:
        while True:
            i = 10
    '\n    Dump pointers and symbols at the specified address.\n    '
    return pwndbg.commands.telescope.telescope(addr)
da_parser = argparse.ArgumentParser(description='Dump a string at the specified address.')
da_parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='Address to dump')
da_parser.add_argument('max', type=int, nargs='?', default=256, help='Maximum string length')

@pwndbg.commands.ArgparsedCommand(da_parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def da(address, max) -> None:
    if False:
        while True:
            i = 10
    print('%x' % address, repr(pwndbg.gdblib.strings.get(address, max)))
ds_parser = argparse.ArgumentParser(description='Dump a string at the specified address.')
ds_parser.add_argument('address', type=pwndbg.commands.HexOrAddressExpr, help='Address to dump')
ds_parser.add_argument('max', type=int, nargs='?', default=256, help='Maximum string length')

@pwndbg.commands.ArgparsedCommand(ds_parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def ds(address, max) -> None:
    if False:
        while True:
            i = 10
    if max < 256:
        print('Max str len of %d too low, changing to 256' % max)
        max = 256
    string = pwndbg.gdblib.strings.get(address, max, maxread=4096)
    if string:
        print(f'{address:x} {string!r}')
    else:
        print("Data at address can't be dereferenced or is not a printable null-terminated string or is too short.")
        print('Perhaps try: db <address> <count> or hexdump <address>')

@pwndbg.commands.ArgparsedCommand('List breakpoints.', category=CommandCategory.WINDBG)
def bl() -> None:
    if False:
        while True:
            i = 10
    '\n    List breakpoints\n    '
    gdb.execute('info breakpoints')
parser = argparse.ArgumentParser(description='Disable the breakpoint with the specified index.')
parser.add_argument('which', nargs='?', type=str, default='*', help='Index of the breakpoint to disable.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
def bd(which='*') -> None:
    if False:
        i = 10
        return i + 15
    '\n    Disable the breakpoint with the specified index.\n    '
    if which == '*':
        gdb.execute('disable breakpoints')
    else:
        gdb.execute(f'disable breakpoints {which}')
parser = argparse.ArgumentParser(description='Enable the breakpoint with the specified index.')
parser.add_argument('which', nargs='?', type=str, default='*', help='Index of the breakpoint to enable.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
def be(which='*') -> None:
    if False:
        while True:
            i = 10
    '\n    Enable the breakpoint with the specified index.\n    '
    if which == '*':
        gdb.execute('enable breakpoints')
    else:
        gdb.execute(f'enable breakpoints {which}')
parser = argparse.ArgumentParser(description='Clear the breakpoint with the specified index.')
parser.add_argument('which', nargs='?', type=str, default='*', help='Index of the breakpoint to clear.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
def bc(which='*') -> None:
    if False:
        return 10
    '\n    Clear the breakpoint with the specified index.\n    '
    if which == '*':
        gdb.execute('delete breakpoints')
    else:
        gdb.execute(f'delete breakpoints {which}')
parser = argparse.ArgumentParser(description='Set a breakpoint at the specified address.')
parser.add_argument('where', type=int, help='The address to break at.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
def bp(where) -> None:
    if False:
        while True:
            i = 10
    '\n    Set a breakpoint at the specified address.\n    '
    result = pwndbg.commands.fix(where)
    if result is not None:
        gdb.execute('break *%#x' % int(result))

@pwndbg.commands.ArgparsedCommand("Print a backtrace (alias 'bt').", category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def k() -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Print a backtrace (alias 'bt')\n    "
    gdb.execute('bt')
parser = argparse.ArgumentParser(description='List the symbols nearest to the provided value.')
parser.add_argument('value', type=int, nargs='?', default=None, help='The address you want the name of.')

@pwndbg.commands.ArgparsedCommand(parser, category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def ln(value=None) -> None:
    if False:
        i = 10
        return i + 15
    '\n    List the symbols nearest to the provided value.\n    '
    if value is None:
        value = pwndbg.gdblib.regs.pc
    value = int(value)
    x = pwndbg.gdblib.symbol.get(value)
    if x:
        result = f'({value:#x})   {x}'
        print(result)

@pwndbg.commands.ArgparsedCommand('Not be windows.', category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def peb() -> None:
    if False:
        print('Hello World!')
    print("This isn't Windows!")

@pwndbg.commands.ArgparsedCommand("Windbg compatibility alias for 'continue' command.", category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def go() -> None:
    if False:
        i = 10
        return i + 15
    "\n    Windbg compatibility alias for 'continue' command.\n    "
    gdb.execute('continue')

@pwndbg.commands.ArgparsedCommand("Windbg compatibility alias for 'nextcall' command.", category=CommandCategory.WINDBG)
@pwndbg.commands.OnlyWhenRunning
def pc():
    if False:
        i = 10
        return i + 15
    "\n    Windbg compatibility alias for 'nextcall' command.\n    "
    return pwndbg.commands.next.nextcall()