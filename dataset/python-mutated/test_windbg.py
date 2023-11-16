from __future__ import annotations
import gdb
import pwndbg
import tests
MEMORY_BINARY = tests.binaries.get('memory.out')
X86_BINARY = tests.binaries.get('gosample.x86')
data_addr = '0x400081'

def test_windbg_dX_commands(start_binary):
    if False:
        print('Hello World!')
    '\n    Tests windbg compatibility commands that dump memory\n    like dq, dw, db, ds etc.\n    '
    start_binary(MEMORY_BINARY)
    for cmd_prefix in ('dq', 'dd', 'dw', 'db'):
        cmd = cmd_prefix + ' nonexistentsymbol'
        assert gdb.execute(cmd, to_string=True) == 'usage: XX [-h] address [count]\nXX: error: argument address: Incorrect address (or GDB expression): nonexistentsymbol\n'.replace('XX', cmd_prefix)
        cmd = cmd_prefix + ' 0'
        assert gdb.execute(cmd, to_string=True) == 'Could not access the provided address\n'
    dq1 = gdb.execute('dq data', to_string=True)
    dq2 = gdb.execute('dq &data', to_string=True)
    dq3 = gdb.execute(f'dq {data_addr}', to_string=True)
    dq4 = gdb.execute(f"dq {data_addr.replace('0x', '')}", to_string=True)
    assert dq1 == dq2 == dq3 == dq4 == '0000000000400081     0000000000000000 0000000000000001\n0000000000400091     0000000100000002 0001000200030004\n00000000004000a1     0102030405060708 1122334455667788\n00000000004000b1     0123456789abcdef 0000000000000000\n'
    dq_count1 = gdb.execute('dq data 2', to_string=True)
    dq_count2 = gdb.execute('dq &data 2', to_string=True)
    dq_count3 = gdb.execute(f'dq {data_addr} 2', to_string=True)
    assert dq_count1 == dq_count2 == dq_count3 == '0000000000400081     0000000000000000 0000000000000001\n'
    assert gdb.execute('dq data 1', to_string=True) == '0000000000400081     0000000000000000\n'
    assert gdb.execute('dq data 3', to_string=True) == '0000000000400081     0000000000000000 0000000000000001\n0000000000400091     0000000100000002\n'
    assert gdb.execute('set $eax=4', to_string=True) == ''
    assert gdb.execute('dq data2 $eax', to_string=True) == '00000000004000a9     1122334455667788 0123456789abcdef\n00000000004000b9     0000000000000000 ffffffffffffffff\n'
    assert gdb.execute('dq data2 2', to_string=True) == '00000000004000a9     1122334455667788 0123456789abcdef\n'
    dd1 = gdb.execute('dd data', to_string=True)
    dd2 = gdb.execute('dd &data', to_string=True)
    dd3 = gdb.execute(f'dd {data_addr}', to_string=True)
    dd4 = gdb.execute(f"dd {data_addr.replace('0x', '')}", to_string=True)
    assert dd1 == dd2 == dd3 == dd4 == '0000000000400081     00000000 00000000 00000001 00000000\n0000000000400091     00000002 00000001 00030004 00010002\n00000000004000a1     05060708 01020304 55667788 11223344\n00000000004000b1     89abcdef 01234567 00000000 00000000\n'
    assert gdb.execute('dd data 4', to_string=True) == '0000000000400081     00000000 00000000 00000001 00000000\n'
    assert gdb.execute('dd data 3', to_string=True) == '0000000000400081     00000000 00000000 00000001\n'
    dw1 = gdb.execute('dw data', to_string=True)
    dw2 = gdb.execute('dw &data', to_string=True)
    dw3 = gdb.execute(f'dw {data_addr}', to_string=True)
    dw4 = gdb.execute(f"dw {data_addr.replace('0x', '')}", to_string=True)
    assert dw1 == dw2 == dw3 == dw4 == '0000000000400081     0000 0000 0000 0000 0001 0000 0000 0000\n0000000000400091     0002 0000 0001 0000 0004 0003 0002 0001\n00000000004000a1     0708 0506 0304 0102 7788 5566 3344 1122\n00000000004000b1     cdef 89ab 4567 0123 0000 0000 0000 0000\n'
    assert gdb.execute('dw data 8', to_string=True) == '0000000000400081     0000 0000 0000 0000 0001 0000 0000 0000\n'
    assert gdb.execute('dw data 8/2', to_string=True) == '0000000000400081     0000 0000 0000 0000\n'
    assert gdb.execute('dw data $eax', to_string=True) == '0000000000400081     0000 0000 0000 0000\n'
    db1 = gdb.execute('db data', to_string=True)
    db2 = gdb.execute('db &data', to_string=True)
    db3 = gdb.execute(f'db {data_addr}', to_string=True)
    db4 = gdb.execute(f"db {data_addr.replace('0x', '')}", to_string=True)
    assert db1 == db2 == db3 == db4 == '0000000000400081     00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00\n0000000000400091     02 00 00 00 01 00 00 00 04 00 03 00 02 00 01 00\n00000000004000a1     08 07 06 05 04 03 02 01 88 77 66 55 44 33 22 11\n00000000004000b1     ef cd ab 89 67 45 23 01 00 00 00 00 00 00 00 00\n'
    assert gdb.execute('db data 31', to_string=True) == '0000000000400081     00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00\n0000000000400091     02 00 00 00 01 00 00 00 04 00 03 00 02 00 01\n'
    assert gdb.execute('db data $ax', to_string=True) == '0000000000400081     00 00 00 00\n'
    dc1 = gdb.execute('dc data', to_string=True)
    dc2 = gdb.execute('dc &data', to_string=True)
    dc3 = gdb.execute(f'dc {data_addr}', to_string=True)
    dc4 = gdb.execute(f"dc {data_addr.replace('0x', '')}", to_string=True)
    assert dc1 == dc2 == dc3 == dc4 == '+0000 0x400081  00 00 00 00 00 00 00 00                           │........│        │\n'
    assert gdb.execute('dc data 3', to_string=True) == '+0000 0x400081  00 00 00                                          │...     │        │\n'
    ds1 = gdb.execute('ds short_str', to_string=True)
    ds2 = gdb.execute('ds &short_str', to_string=True)
    ds3 = gdb.execute('ds 0x4000d9', to_string=True)
    ds4 = gdb.execute('ds 4000d9', to_string=True)
    assert ds1 == ds2 == ds3 == ds4 == "4000d9 'some cstring here'\n"
    assert gdb.execute('ds short_str 5', to_string=True) == "Max str len of 5 too low, changing to 256\n4000d9 'some cstring here'\n"
    assert gdb.execute('ds long_str', to_string=True) == "4000eb 'long string: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA...'\n"
    assert gdb.execute('ds 0', to_string=True) == "Data at address can't be dereferenced or is not a printable null-terminated string or is too short.\nPerhaps try: db <address> <count> or hexdump <address>\n"

def test_windbg_eX_commands(start_binary):
    if False:
        i = 10
        return i + 15
    '\n    Tests windbg compatibility commands that write to memory\n    like eq, ed, ew, eb etc.\n    '
    start_binary(MEMORY_BINARY)
    for cmd_prefix in ('eq', 'ed', 'ew', 'eb'):
        cmd = cmd_prefix + ' nonexistentsymbol'
        expected_in = ('usage: XX [-h] address [data ...]\nXX: error: argument address: Incorrect address (or GDB expression): nonexistentsymbol\n'.replace('XX', cmd_prefix), 'usage: XX [-h] address [data [data ...]]\nXX: error: argument address: Incorrect address (or GDB expression): nonexistentsymbol\n'.replace('XX', cmd_prefix))
        assert gdb.execute(cmd, to_string=True) in expected_in
        assert gdb.execute(cmd, to_string=True) in expected_in
        cmd = cmd_prefix + ' 0'
        assert gdb.execute(cmd, to_string=True) == 'Cannot write empty data into memory.\n'
        cmd = cmd_prefix + ' 0 1122'
        assert gdb.execute(cmd, to_string=True) == 'Cannot access memory at address 0x0\n'
        cmd = cmd_prefix + ' 0 x'
        assert gdb.execute(cmd, to_string=True) == 'Incorrect data format: it must all be a hex value (0x1234 or 1234, both interpreted as 0x1234)\n'
    assert gdb.execute('eq $sp 0xcafebabe', to_string=True) == ''
    assert '0x00000000cafebabe' in gdb.execute('x/xg $sp', to_string=True)
    assert gdb.execute('eq $sp 0xbabe 0xcafe', to_string=True) == ''
    assert '0x000000000000babe\t0x000000000000cafe' in gdb.execute('x/2xg $sp', to_string=True)
    assert gdb.execute('eq $sp cafe000000000000 babe000000000000', to_string=True) == ''
    assert '0xcafe000000000000\t0xbabe000000000000' in gdb.execute('x/2xg $sp', to_string=True)
    stack_ea = pwndbg.gdblib.regs[pwndbg.gdblib.regs.stack]
    stack_page = pwndbg.gdblib.vmmap.find(stack_ea)
    stack_last_qword_ea = stack_page.end - 8
    gdb_result = gdb.execute('eq %#x 0xCAFEBABEdeadbeef 0xABCD' % stack_last_qword_ea, to_string=True).split('\n')
    assert 'Cannot access memory at address' in gdb_result[0]
    assert gdb_result[1] == '(Made 1 writes to memory; skipping further writes)'
    assert pwndbg.gdblib.memory.read(stack_last_qword_ea, 8) == b'\xef\xbe\xad\xde\xbe\xba\xfe\xca'

def test_windbg_commands_x86(start_binary):
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests windbg compatibility commands that dump memory\n    like dq, dw, db, ds etc.\n    '
    start_binary(X86_BINARY)
    pwndbg.gdblib.memory.write(pwndbg.gdblib.regs.esp, b'1234567890abcdef_')
    pwndbg.gdblib.memory.write(pwndbg.gdblib.regs.esp + 16, b'\x00' * 16)
    pwndbg.gdblib.memory.write(pwndbg.gdblib.regs.esp + 32, bytes(range(16)))
    pwndbg.gdblib.memory.write(pwndbg.gdblib.regs.esp + 48, b'Z' * 16)
    db = gdb.execute('db $esp', to_string=True).splitlines()
    assert db == ['%x     31 32 33 34 35 36 37 38 39 30 61 62 63 64 65 66' % pwndbg.gdblib.regs.esp, '%x     00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00' % (pwndbg.gdblib.regs.esp + 16), '%x     00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f' % (pwndbg.gdblib.regs.esp + 32), '%x     5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a' % (pwndbg.gdblib.regs.esp + 48)]
    dw = gdb.execute('dw $esp', to_string=True).splitlines()
    assert dw == ['%x     3231 3433 3635 3837 3039 6261 6463 6665' % pwndbg.gdblib.regs.esp, '%x     0000 0000 0000 0000 0000 0000 0000 0000' % (pwndbg.gdblib.regs.esp + 16), '%x     0100 0302 0504 0706 0908 0b0a 0d0c 0f0e' % (pwndbg.gdblib.regs.esp + 32), '%x     5a5a 5a5a 5a5a 5a5a 5a5a 5a5a 5a5a 5a5a' % (pwndbg.gdblib.regs.esp + 48)]
    dd = gdb.execute('dd $esp', to_string=True).splitlines()
    assert dd == ['%x     34333231 38373635 62613039 66656463' % pwndbg.gdblib.regs.esp, '%x     00000000 00000000 00000000 00000000' % (pwndbg.gdblib.regs.esp + 16), '%x     03020100 07060504 0b0a0908 0f0e0d0c' % (pwndbg.gdblib.regs.esp + 32), '%x     5a5a5a5a 5a5a5a5a 5a5a5a5a 5a5a5a5a' % (pwndbg.gdblib.regs.esp + 48)]
    dq = gdb.execute('dq $esp', to_string=True).splitlines()
    assert dq == ['%x     3837363534333231 6665646362613039' % pwndbg.gdblib.regs.esp, '%x     0000000000000000 0000000000000000' % (pwndbg.gdblib.regs.esp + 16), '%x     0706050403020100 0f0e0d0c0b0a0908' % (pwndbg.gdblib.regs.esp + 32), '%x     5a5a5a5a5a5a5a5a 5a5a5a5a5a5a5a5a' % (pwndbg.gdblib.regs.esp + 48)]
    gdb.execute('eb $esp 00')
    assert pwndbg.gdblib.memory.read(pwndbg.gdblib.regs.esp, 1) == b'\x00'
    gdb.execute('ew $esp 4141')
    assert pwndbg.gdblib.memory.read(pwndbg.gdblib.regs.esp, 2) == b'AA'
    gdb.execute('ed $esp 5252525252')
    assert pwndbg.gdblib.memory.read(pwndbg.gdblib.regs.esp, 4) == b'R' * 4
    gdb.execute('eq $esp 1122334455667788')
    assert pwndbg.gdblib.memory.read(pwndbg.gdblib.regs.esp, 8) == b'\x88wfUD3"\x11'