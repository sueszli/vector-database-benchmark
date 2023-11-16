from __future__ import print_function
import struct
from sys import stdout
try:
    stdout = stdout.buffer
except AttributeError:
    pass
from miasm.core.utils import int_to_byte, cmp_elts
from miasm.os_dep.common import heap
from miasm.os_dep.common import get_fmt_args as _get_fmt_args

class c_linobjs(object):
    base_addr = 536870912
    align_addr = 4096

    def __init__(self):
        if False:
            while True:
                i = 10
        self.alloc_ad = self.base_addr
        self.alloc_align = self.align_addr
        self.heap = heap()
linobjs = c_linobjs()
ABORT_ADDR = 322420463

def xxx___libc_start_main(jitter):
    if False:
        i = 10
        return i + 15
    'Basic implementation of __libc_start_main\n\n    int __libc_start_main(int *(main) (int, char * *, char * *), int argc,\n                          char * * ubp_av, void (*init) (void),\n                          void (*fini) (void), void (*rtld_fini) (void),\n                          void (* stack_end));\n\n    Note:\n     - init, fini, rtld_fini are ignored\n     - return address is forced to ABORT_ADDR, to avoid calling abort/hlt/...\n     - in powerpc, signature is:\n\n    int __libc_start_main (int argc, char **argv, char **ev, ElfW (auxv_t) *\n                       auxvec, void (*rtld_fini) (void), struct startup_info\n                       *stinfo, char **stack_on_entry)\n\n    '
    global ABORT_ADDR
    if jitter.arch.name == 'ppc32':
        (ret_ad, args) = jitter.func_args_systemv(['argc', 'argv', 'ev', 'aux_vec', 'rtld_fini', 'st_info', 'stack_on_entry'])
        if args.stack_on_entry != 0:
            argc = struct.unpack('>I', jitter.vm.get_mem(args.stack_on_entry, 4))[0]
            argv = args.stack_on_entry + 4
            envp = argv + (argc + 1) * 4
        else:
            argc = args.argc
            argv = args.argv
            envp = args.ev
        (_, main, _, _) = struct.unpack('>IIII', jitter.vm.get_mem(args.st_info, 4 * 4))
    else:
        (ret_ad, args) = jitter.func_args_systemv(['main', 'argc', 'ubp_av', 'init', 'fini', 'rtld_fini', 'stack_end'])
        main = args.main
        size = jitter.lifter.pc.size // 8
        argc = args.argc
        argv = args.ubp_av
        envp = argv + (args.argc + 1) * size
    jitter.func_ret_systemv(main)
    ret_ad = ABORT_ADDR
    jitter.func_prepare_systemv(ret_ad, argc, argv, envp)
    return True

def xxx_isprint(jitter):
    if False:
        return 10
    '\n    #include <ctype.h>\n    int isprint(int c);\n\n    checks for any printable character including space.\n    '
    (ret_addr, args) = jitter.func_args_systemv(['c'])
    ret = 1 if 32 <= args.c & 255 < 127 else 0
    return jitter.func_ret_systemv(ret_addr, ret)

def xxx_memcpy(jitter):
    if False:
        print('Hello World!')
    '\n    #include <string.h>\n    void *memcpy(void *dest, const void *src, size_t n);\n\n    copies n bytes from memory area src to memory area dest.\n    '
    (ret_addr, args) = jitter.func_args_systemv(['dest', 'src', 'n'])
    jitter.vm.set_mem(args.dest, jitter.vm.get_mem(args.src, args.n))
    return jitter.func_ret_systemv(ret_addr, args.dest)

def xxx_memset(jitter):
    if False:
        i = 10
        return i + 15
    '\n    #include <string.h>\n    void *memset(void *s, int c, size_t n);\n\n    fills the first n bytes of the memory area pointed to by s with the constant\n    byte c.'
    (ret_addr, args) = jitter.func_args_systemv(['dest', 'c', 'n'])
    jitter.vm.set_mem(args.dest, int_to_byte(args.c & 255) * args.n)
    return jitter.func_ret_systemv(ret_addr, args.dest)

def xxx_puts(jitter):
    if False:
        i = 10
        return i + 15
    '\n    #include <stdio.h>\n    int puts(const char *s);\n\n    writes the string s and a trailing newline to stdout.\n    '
    (ret_addr, args) = jitter.func_args_systemv(['s'])
    index = args.s
    char = jitter.vm.get_mem(index, 1)
    while char != b'\x00':
        stdout.write(char)
        index += 1
        char = jitter.vm.get_mem(index, 1)
    stdout.write(b'\n')
    return jitter.func_ret_systemv(ret_addr, 1)

def get_fmt_args(jitter, fmt, cur_arg):
    if False:
        return 10
    return _get_fmt_args(fmt, cur_arg, jitter.get_c_str, jitter.get_arg_n_systemv)

def xxx_snprintf(jitter):
    if False:
        while True:
            i = 10
    (ret_addr, args) = jitter.func_args_systemv(['string', 'size', 'fmt'])
    (cur_arg, fmt) = (3, args.fmt)
    size = args.size if args.size else 1
    output = get_fmt_args(jitter, fmt, cur_arg)
    output = output[:size - 1]
    ret = len(output)
    jitter.set_c_str(args.string, output)
    return jitter.func_ret_systemv(ret_addr, ret)

def xxx_sprintf(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_addr, args) = jitter.func_args_systemv(['string', 'fmt'])
    (cur_arg, fmt) = (2, args.fmt)
    output = get_fmt_args(jitter, fmt, cur_arg)
    ret = len(output)
    jitter.set_c_str(args.string, output)
    return jitter.func_ret_systemv(ret_addr, ret)

def xxx_printf(jitter):
    if False:
        i = 10
        return i + 15
    (ret_addr, args) = jitter.func_args_systemv(['fmt'])
    (cur_arg, fmt) = (1, args.fmt)
    output = get_fmt_args(jitter, fmt, cur_arg)
    ret = len(output)
    stdout.write(output.encode('utf8'))
    return jitter.func_ret_systemv(ret_addr, ret)

def xxx_strcpy(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_systemv(['dst', 'src'])
    str_src = jitter.get_c_str(args.src)
    jitter.set_c_str(args.dst, str_src)
    jitter.func_ret_systemv(ret_ad, args.dst)

def xxx_strlen(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_systemv(['src'])
    str_src = jitter.get_c_str(args.src)
    jitter.func_ret_systemv(ret_ad, len(str_src))

def xxx_malloc(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_systemv(['msize'])
    addr = linobjs.heap.alloc(jitter, args.msize)
    jitter.func_ret_systemv(ret_ad, addr)

def xxx_free(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_systemv(['ptr'])
    jitter.func_ret_systemv(ret_ad, 0)

def xxx_strcmp(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_systemv(['ptr_str1', 'ptr_str2'])
    s1 = jitter.get_c_str(args.ptr_str1)
    s2 = jitter.get_c_str(args.ptr_str2)
    jitter.func_ret_systemv(ret_ad, cmp_elts(s1, s2))

def xxx_strncmp(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_systemv(['ptr_str1', 'ptr_str2', 'size'])
    s1 = jitter.get_c_str(args.ptr_str1, args.size)
    s2 = jitter.get_c_str(args.ptr_str2, args.size)
    jitter.func_ret_systemv(ret_ad, cmp_elts(s1, s2))