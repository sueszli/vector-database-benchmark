from __future__ import annotations
import re
import gdb
import pytest
import pwndbg.commands
import tests
USE_FDS_BINARY = tests.binaries.get('use-fds.out')
TABSTOP_BINARY = tests.binaries.get('tabstop.out')
SYSCALLS_BINARY = tests.binaries.get('syscalls-x64.out')
MANGLING_BINARY = tests.binaries.get('symbol_1600_and_752.out')

def test_context_disasm_show_fd_filepath(start_binary):
    if False:
        i = 10
        return i + 15
    '\n    Tests context disasm command and whether it shows properly opened fd filepath\n    '
    start_binary(USE_FDS_BINARY)
    gdb.execute('break main')
    gdb.execute('continue')
    gdb.execute('nextcall')
    out = pwndbg.commands.context.context_disasm()
    assert '[ DISASM / x86-64 / set emulate on ]' in out[0]
    call_read_line_idx = out.index(next((line for line in out if '<read@plt>' in line)))
    lines_after_call_read = out[call_read_line_idx:]
    (line_call_read, line_fd, line_buf, line_nbytes, *_rest) = lines_after_call_read
    assert 'call   read@plt' in line_call_read
    line_fd = line_fd.strip()
    assert re.match('fd:\\s+0x1 \\((/dev/pts/\\d+|/tmp/par.+\\.par(?: \\(deleted\\))?|pipe:\\[\\d+\\])\\)', line_fd)
    line_buf = line_buf.strip()
    assert re.match('buf:\\s+0x[0-9a-f]+ ◂— 0x0', line_buf)
    line_nbytes = line_nbytes.strip()
    assert re.match('nbytes:\\s+0x0', line_nbytes)
    gdb.execute('nextcall')
    gdb.execute('nextcall')
    out = pwndbg.commands.context.context_disasm()
    assert '[ DISASM / x86-64 / set emulate on ]' in out[0]
    call_read_line_idx = out.index(next((line for line in out if '<read@plt>' in line)))
    lines_after_call_read = out[call_read_line_idx:]
    (line_call_read, line_fd, line_buf, line_nbytes, *_rest) = lines_after_call_read
    line_fd = line_fd.strip()
    assert re.match('fd:\\s+0x3 \\([a-z/]*pwndbg/tests/gdb-tests/tests/binaries/use-fds.out\\)', line_fd)
    line_buf = line_buf.strip()
    assert re.match('buf:\\s+0x[0-9a-f]+ ◂— 0x0', line_buf)
    line_nbytes = line_nbytes.strip()
    assert re.match('nbytes:\\s+0x10', line_nbytes)

@pytest.mark.parametrize('sections', ("''", '""', 'none', '-', ''))
def test_empty_context_sections(start_binary, sections):
    if False:
        print('Hello World!')
    start_binary(USE_FDS_BINARY)
    default_ctx_sects = 'regs disasm code ghidra stack backtrace expressions threads'
    assert pwndbg.gdblib.config.context_sections.value == default_ctx_sects
    assert gdb.execute('context', to_string=True) != ''
    gdb.execute(f'set context-sections {sections}', to_string=True)
    assert pwndbg.gdblib.config.context_sections.value == ''
    assert gdb.execute('context', to_string=True) == ''
    gdb.execute(f'set context-sections {default_ctx_sects}')
    assert pwndbg.gdblib.config.context_sections.value == default_ctx_sects
    assert gdb.execute('context', to_string=True) != ''

def test_source_code_tabstop(start_binary):
    if False:
        while True:
            i = 10
    start_binary(TABSTOP_BINARY)
    gdb.execute('break tabstop.c:6')
    gdb.execute('continue')
    src = gdb.execute('context code', to_string=True)
    assert ' 1 #include <stdio.h>\n' in src
    assert ' 2 \n' in src
    assert ' 3 int main() {\n' in src
    assert ' 4         // test mix indent\n' in src
    assert ' 5         do {\n' in src
    assert ' 6                 puts("tab line");\n' in src
    assert ' 7         } while (0);\n' in src
    assert ' 8         return 0;\n' in src
    assert ' 9 }\n' in src
    assert '10 \n' in src
    gdb.execute('set context-source-code-tabstop 2')
    src = gdb.execute('context code', to_string=True)
    assert ' 1 #include <stdio.h>\n' in src
    assert ' 2 \n' in src
    assert ' 3 int main() {\n' in src
    assert ' 4   // test mix indent\n' in src
    assert ' 5         do {\n' in src
    assert ' 6     puts("tab line");\n' in src
    assert ' 7         } while (0);\n' in src
    assert ' 8         return 0;\n' in src
    assert ' 9 }\n' in src
    assert '10 \n' in src
    gdb.execute('set context-source-code-tabstop 0')
    src = gdb.execute('context code', to_string=True)
    assert ' 1 #include <stdio.h>\n' in src
    assert ' 2 \n' in src
    assert ' 3 int main() {\n' in src
    assert ' 4 \t// test mix indent\n' in src
    assert ' 5         do {\n' in src
    assert ' 6 \t\tputs("tab line");\n' in src
    assert ' 7         } while (0);\n' in src
    assert ' 8         return 0;\n' in src
    assert ' 9 }\n' in src
    assert '10 \n' in src

def test_context_disasm_syscalls_args_display(start_binary):
    if False:
        for i in range(10):
            print('nop')
    start_binary(SYSCALLS_BINARY)
    gdb.execute('nextsyscall')
    dis = gdb.execute('context disasm', to_string=True)
    assert dis == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA\n──────────────────────[ DISASM / x86-64 / set emulate on ]──────────────────────\n   0x400080 <_start>       mov    eax, 0\n   0x400085 <_start+5>     mov    edi, 0x1337\n   0x40008a <_start+10>    mov    esi, 0xdeadbeef\n   0x40008f <_start+15>    mov    ecx, 0x10\n ► 0x400094 <_start+20>    syscall  <SYS_read>\n        fd:        0x1337\n        buf:       0xdeadbeef\n        nbytes:    0x0\n   0x400096 <_start+22>    mov    eax, 0xa\n   0x40009b <_start+27>    int    0x80\n   0x40009d                add    byte ptr [rax], al\n   0x40009f                add    byte ptr [rax], al\n   0x4000a1                add    byte ptr [rax], al\n   0x4000a3                add    byte ptr [rax], al\n────────────────────────────────────────────────────────────────────────────────\n'
    gdb.execute('nextsyscall')
    dis = gdb.execute('context disasm', to_string=True)
    assert dis == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA\n──────────────────────[ DISASM / x86-64 / set emulate on ]──────────────────────\n   0x400085 <_start+5>     mov    edi, 0x1337\n   0x40008a <_start+10>    mov    esi, 0xdeadbeef\n   0x40008f <_start+15>    mov    ecx, 0x10\n   0x400094 <_start+20>    syscall \n   0x400096 <_start+22>    mov    eax, 0xa\n ► 0x40009b <_start+27>    int    0x80 <SYS_unlink>\n        name:      0x1337\n   0x40009d                add    byte ptr [rax], al\n   0x40009f                add    byte ptr [rax], al\n   0x4000a1                add    byte ptr [rax], al\n   0x4000a3                add    byte ptr [rax], al\n   0x4000a5                add    byte ptr [rax], al\n────────────────────────────────────────────────────────────────────────────────\n'

def test_context_backtrace_show_proper_symbol_names(start_binary):
    if False:
        return 10
    start_binary(MANGLING_BINARY)
    gdb.execute('break A::foo')
    gdb.execute('continue')
    backtrace = gdb.execute('context backtrace', to_string=True).split('\n')
    assert backtrace[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    assert backtrace[1] == '─────────────────────────────────[ BACKTRACE ]──────────────────────────────────'
    assert re.match('.*0   0x[0-9a-f]+ A::foo\\(int, int\\)', backtrace[2])
    assert re.match('.*1   0x[0-9a-f]+ A::call_foo\\(\\)\\+\\d+', backtrace[3])
    assert re.match('.*2   0x[0-9a-f]+ main\\+\\d+', backtrace[4])
    assert re.match('.*3   0x[0-9a-f]+ (__libc_start_main|__libc_start_call_main)\\+\\d+', backtrace[5])
    assert backtrace[-2] == '────────────────────────────────────────────────────────────────────────────────'
    assert backtrace[-1] == ''

def test_context_disasm_works_properly_with_disasm_flavor_switch(start_binary):
    if False:
        i = 10
        return i + 15
    start_binary(SYSCALLS_BINARY)

    def assert_intel(out):
        if False:
            return 10
        assert 'mov    eax, 0' in out[2]
        assert 'mov    edi, 0x1337' in out[3]
        assert 'mov    esi, 0xdeadbeef' in out[4]
        assert 'mov    ecx, 0x10' in out[5]
        assert 'syscall' in out[6]

    def assert_att(out):
        if False:
            while True:
                i = 10
        assert 'mov    movl   $0, %eax' not in out[2]
        assert 'mov    movl   $0x1337, %edi' not in out[3]
        assert 'mov    movl   $0xdeadbeef, %esi' not in out[4]
        assert 'mov    movl   $0x10, %ecx' not in out[5]
        assert 'syscall' in out[6]
    out = gdb.execute('context disasm', to_string=True).split('\n')
    assert out[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    assert out[1] == '──────────────────────[ DISASM / x86-64 / set emulate on ]──────────────────────'
    assert_intel(out)
    gdb.execute('set disassembly-flavor att')
    assert out[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    assert out[1] == '──────────────────────[ DISASM / x86-64 / set emulate on ]──────────────────────'
    assert_att(out)

@pytest.mark.parametrize('patch_or_api', (True, False))
def test_context_disasm_proper_render_on_mem_change_issue_1818(start_binary, patch_or_api):
    if False:
        i = 10
        return i + 15
    start_binary(SYSCALLS_BINARY)
    old = gdb.execute('context disasm', to_string=True).split('\n')
    assert old[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    assert 'mov    eax, 0' in old[2]
    assert 'mov    edi, 0x1337' in old[3]
    assert 'mov    esi, 0xdeadbeef' in old[4]
    assert 'mov    ecx, 0x10' in old[5]
    assert 'syscall' in old[6]
    if patch_or_api:
        gdb.execute('patch $rip nop;nop;nop;nop;nop', to_string=True)
    else:
        pwndbg.gdblib.memory.write(pwndbg.gdblib.regs.rip, b'\x90' * 5)
    new = gdb.execute('context disasm', to_string=True).split('\n')
    assert new[0] == 'LEGEND: STACK | HEAP | CODE | DATA | RWX | RODATA'
    assert 'nop' in new[2]
    assert 'nop' in new[3]
    assert 'nop' in new[4]
    assert 'nop' in new[5]
    assert 'nop' in new[6]
    assert 'mov    edi, 0x1337' in new[7]
    assert 'mov    esi, 0xdeadbeef' in new[8]
    assert 'mov    ecx, 0x10' in new[9]
    assert 'syscall' in new[10]