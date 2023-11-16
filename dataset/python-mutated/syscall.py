from builtins import range
import fcntl
import functools
import logging
import struct
import termios
from miasm.jitter.csts import EXCEPT_INT_XX, EXCEPT_SYSCALL
from miasm.core.utils import pck64
log = logging.getLogger('syscalls')
hnd = logging.StreamHandler()
hnd.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log.addHandler(hnd)
log.setLevel(logging.WARNING)

def _dump_struct_stat_x86_64(info):
    if False:
        return 10
    data = struct.pack('QQQIIIIQQQQQQQQQQQQQ', info.st_dev, info.st_ino, info.st_nlink, info.st_mode, info.st_uid, info.st_gid, 0, info.st_rdev, info.st_size, info.st_blksize, info.st_blocks, info.st_atime, info.st_atimensec, info.st_mtime, info.st_mtimensec, info.st_ctime, info.st_ctimensec, 0, 0, 0)
    return data

def _dump_struct_stat_arml(info):
    if False:
        print('Hello World!')
    data = struct.pack('QIIIIIIIIIIIIIIIIII', info.st_dev, 0, info.st_ino, info.st_mode, info.st_nlink, info.st_uid, info.st_gid, info.st_rdev, info.st_size, info.st_blksize, info.st_blocks, info.st_atime, info.st_atimensec, info.st_mtime, info.st_mtimensec, info.st_ctime, info.st_ctimensec, 0, 0)
    return data

def sys_x86_64_rt_sigaction(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (sig, act, oact, sigsetsize) = jitter.syscall_args_systemv(4)
    log.debug('sys_rt_sigaction(%x, %x, %x, %x)', sig, act, oact, sigsetsize)
    if oact != 0:
        jitter.vm.set_mem(oact, b'\x00' * sigsetsize)
    jitter.syscall_ret_systemv(0)

def sys_generic_brk(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    (addr,) = jitter.syscall_args_systemv(1)
    log.debug('sys_brk(%d)', addr)
    jitter.syscall_ret_systemv(linux_env.brk(addr, jitter.vm))

def sys_x86_32_newuname(jitter, linux_env):
    if False:
        print('Hello World!')
    (nameptr,) = jitter.syscall_args_systemv(1)
    log.debug('sys_newuname(%x)', nameptr)
    info = [linux_env.sys_sysname, linux_env.sys_nodename, linux_env.sys_release, linux_env.sys_version, linux_env.sys_machine]
    output = b''
    for elem in info:
        output += elem
        output += b'\x00' * (65 - len(elem))
    jitter.vm.set_mem(nameptr, output)
    jitter.syscall_ret_systemv(0)

def sys_x86_64_newuname(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (nameptr,) = jitter.syscall_args_systemv(1)
    log.debug('sys_newuname(%x)', nameptr)
    info = [linux_env.sys_sysname, linux_env.sys_nodename, linux_env.sys_release, linux_env.sys_version, linux_env.sys_machine]
    output = b''
    for elem in info:
        output += elem
        output += b'\x00' * (65 - len(elem))
    jitter.vm.set_mem(nameptr, output)
    jitter.syscall_ret_systemv(0)

def sys_arml_newuname(jitter, linux_env):
    if False:
        return 10
    (nameptr,) = jitter.syscall_args_systemv(1)
    log.debug('sys_newuname(%x)', nameptr)
    info = [linux_env.sys_sysname, linux_env.sys_nodename, linux_env.sys_release, linux_env.sys_version, linux_env.sys_machine]
    output = b''
    for elem in info:
        output += elem
        output += b'\x00' * (65 - len(elem))
    jitter.vm.set_mem(nameptr, output)
    jitter.syscall_ret_systemv(0)

def sys_generic_access(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (pathname, mode) = jitter.syscall_args_systemv(2)
    rpathname = jitter.get_c_str(pathname)
    rmode = mode
    if mode == 1:
        rmode = 'F_OK'
    elif mode == 2:
        rmode = 'R_OK'
    log.debug('sys_access(%s, %s)', rpathname, rmode)
    if linux_env.filesystem.exists(rpathname):
        jitter.syscall_ret_systemv(0)
    else:
        jitter.syscall_ret_systemv(-1)

def sys_x86_64_openat(jitter, linux_env):
    if False:
        print('Hello World!')
    (dfd, filename, flags, mode) = jitter.syscall_args_systemv(4)
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_openat(%x, %r, %x, %x)', dfd, rpathname, flags, mode)
    jitter.syscall_ret_systemv(linux_env.open_(rpathname, flags))

def sys_x86_64_newstat(jitter, linux_env):
    if False:
        while True:
            i = 10
    (filename, statbuf) = jitter.syscall_args_systemv(2)
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_newstat(%r, %x)', rpathname, statbuf)
    if linux_env.filesystem.exists(rpathname):
        info = linux_env.stat(rpathname)
        data = _dump_struct_stat_x86_64(info)
        jitter.vm.set_mem(statbuf, data)
        jitter.syscall_ret_systemv(0)
    else:
        jitter.syscall_ret_systemv(-1)

def sys_arml_stat64(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (filename, statbuf) = jitter.syscall_args_systemv(2)
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_newstat(%r, %x)', rpathname, statbuf)
    if linux_env.filesystem.exists(rpathname):
        info = linux_env.stat(rpathname)
        data = _dump_struct_stat_arml(info)
        jitter.vm.set_mem(statbuf, data)
        jitter.syscall_ret_systemv(0)
    else:
        jitter.syscall_ret_systemv(-1)

def sys_x86_64_writev(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    (fd, vec, vlen) = jitter.syscall_args_systemv(3)
    log.debug('sys_writev(%d, %d, %x)', fd, vec, vlen)
    fdesc = linux_env.file_descriptors[fd]
    for iovec_num in range(vlen):
        iovec = jitter.vm.get_mem(vec + iovec_num * 8 * 2, 8 * 2)
        (iov_base, iov_len) = struct.unpack('QQ', iovec)
        fdesc.write(jitter.get_c_str(iov_base)[:iov_len])
    jitter.syscall_ret_systemv(vlen)

def sys_arml_writev(jitter, linux_env):
    if False:
        while True:
            i = 10
    (fd, vec, vlen) = jitter.syscall_args_systemv(3)
    log.debug('sys_writev(%d, %d, %x)', fd, vec, vlen)
    fdesc = linux_env.file_descriptors[fd]
    for iovec_num in range(vlen):
        iovec = jitter.vm.get_mem(vec + iovec_num * 4 * 2, 4 * 2)
        (iov_base, iov_len) = struct.unpack('II', iovec)
        fdesc.write(jitter.get_c_str(iov_base)[:iov_len])
    jitter.syscall_ret_systemv(vlen)

def sys_generic_exit_group(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    (status,) = jitter.syscall_args_systemv(1)
    log.debug('sys_exit_group(%d)', status)
    log.debug('Exit with status code %d', status)
    jitter.running = False

def sys_generic_read(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    (fd, buf, count) = jitter.syscall_args_systemv(3)
    log.debug('sys_read(%d, %x, %x)', fd, buf, count)
    data = linux_env.read(fd, count)
    jitter.vm.set_mem(buf, data)
    jitter.syscall_ret_systemv(len(data))

def sys_x86_64_fstat(jitter, linux_env):
    if False:
        while True:
            i = 10
    (fd, statbuf) = jitter.syscall_args_systemv(2)
    log.debug('sys_fstat(%d, %x)', fd, statbuf)
    info = linux_env.fstat(fd)
    data = _dump_struct_stat_x86_64(info)
    jitter.vm.set_mem(statbuf, data)
    jitter.syscall_ret_systemv(0)

def sys_arml_fstat64(jitter, linux_env):
    if False:
        return 10
    (fd, statbuf) = jitter.syscall_args_systemv(2)
    log.debug('sys_fstat(%d, %x)', fd, statbuf)
    info = linux_env.fstat(fd)
    data = _dump_struct_stat_arml(info)
    jitter.vm.set_mem(statbuf, data)
    jitter.syscall_ret_systemv(0)

def sys_generic_mmap(jitter, linux_env):
    if False:
        while True:
            i = 10
    (addr, len_, prot, flags, fd, off) = jitter.syscall_args_systemv(6)
    log.debug('sys_mmap(%x, %x, %x, %x, %x, %x)', addr, len_, prot, flags, fd, off)
    addr = linux_env.mmap(addr, len_, prot & 4294967295, flags & 4294967295, fd & 4294967295, off, jitter.vm)
    jitter.syscall_ret_systemv(addr)

def sys_generic_mmap2(jitter, linux_env):
    if False:
        while True:
            i = 10
    (addr, len_, prot, flags, fd, off) = jitter.syscall_args_systemv(6)
    log.debug('sys_mmap2(%x, %x, %x, %x, %x, %x)', addr, len_, prot, flags, fd, off)
    off = off * 4096
    addr = linux_env.mmap(addr, len_, prot & 4294967295, flags & 4294967295, fd & 4294967295, off, jitter.vm)
    jitter.syscall_ret_systemv(addr)

def sys_generic_mprotect(jitter, linux_env):
    if False:
        print('Hello World!')
    (start, len_, prot) = jitter.syscall_args_systemv(3)
    assert jitter.vm.is_mapped(start, len_)
    log.debug('sys_mprotect(%x, %x, %x)', start, len_, prot)
    jitter.syscall_ret_systemv(0)

def sys_generic_close(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (fd,) = jitter.syscall_args_systemv(1)
    log.debug('sys_close(%x)', fd)
    linux_env.close(fd)
    jitter.syscall_ret_systemv(0)

def sys_x86_64_arch_prctl(jitter, linux_env):
    if False:
        while True:
            i = 10
    code_name = {4097: 'ARCH_SET_GS', 4098: 'ARCH_SET_FS', 4099: 'ARCH_GET_FS', 4100: 'ARCH_GET_GS', 4113: 'ARCH_GET_CPUID', 4114: 'ARCH_SET_CPUID', 8193: 'ARCH_MAP_VDSO_X32', 8194: 'ARCH_MAP_VDSO_32', 8195: 'ARCH_MAP_VDSO_64', 12289: 'ARCH_CET_STATUS', 12290: 'ARCH_CET_DISABLE', 12291: 'ARCH_CET_LOCK', 12292: 'ARCH_CET_EXEC', 12293: 'ARCH_CET_ALLOC_SHSTK', 12294: 'ARCH_CET_PUSH_SHSTK', 12295: 'ARCH_CET_LEGACY_BITMAP'}
    code = jitter.cpu.RDI
    rcode = code_name[code]
    addr = jitter.cpu.RSI
    log.debug('sys_arch_prctl(%s, %x)', rcode, addr)
    if code == 4098:
        jitter.cpu.set_segm_base(jitter.cpu.FS, addr)
    elif code == 12289:
        jitter.vm.set_mem(addr, pck64(0))
    else:
        raise RuntimeError('Not implemented')
    jitter.cpu.RAX = 0

def sys_x86_64_set_tid_address(jitter, linux_env):
    if False:
        while True:
            i = 10
    tidptr = jitter.cpu.RDI
    log.debug('sys_set_tid_address(%x)', tidptr)
    jitter.cpu.RAX = linux_env.process_tid

def sys_x86_64_set_robust_list(jitter, linux_env):
    if False:
        return 10
    head = jitter.cpu.RDI
    len_ = jitter.cpu.RSI
    log.debug('sys_set_robust_list(%x, %x)', head, len_)
    jitter.cpu.RAX = 0

def sys_x86_64_rt_sigprocmask(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    how = jitter.cpu.RDI
    nset = jitter.cpu.RSI
    oset = jitter.cpu.RDX
    sigsetsize = jitter.cpu.R10
    log.debug('sys_rt_sigprocmask(%x, %x, %x, %x)', how, nset, oset, sigsetsize)
    if oset != 0:
        raise RuntimeError('Not implemented')
    jitter.cpu.RAX = 0

def sys_x86_64_prlimit64(jitter, linux_env):
    if False:
        return 10
    pid = jitter.cpu.RDI
    resource = jitter.cpu.RSI
    new_rlim = jitter.cpu.RDX
    if new_rlim != 0:
        raise RuntimeError('Not implemented')
    old_rlim = jitter.cpu.R10
    log.debug('sys_prlimit64(%x, %x, %x, %x)', pid, resource, new_rlim, old_rlim)
    if resource == 3:
        jitter.vm.set_mem(old_rlim, struct.pack('QQ', 1048576, 9223372036854775807))
    else:
        raise RuntimeError('Not implemented')
    jitter.cpu.RAX = 0

def sys_x86_64_statfs(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    pathname = jitter.cpu.RDI
    buf = jitter.cpu.RSI
    rpathname = jitter.get_c_str(pathname)
    log.debug('sys_statfs(%r, %x)', rpathname, buf)
    if not linux_env.filesystem.exists(rpathname):
        jitter.cpu.RAX = -1
    else:
        info = linux_env.filesystem.statfs()
        raise RuntimeError('Not implemented')

def sys_x86_64_ioctl(jitter, linux_env):
    if False:
        return 10
    (fd, cmd, arg) = jitter.syscall_args_systemv(3)
    log.debug('sys_ioctl(%x, %x, %x)', fd, cmd, arg)
    info = linux_env.ioctl(fd, cmd, arg)
    if info is False:
        jitter.syscall_ret_systemv(-1)
    else:
        if cmd == termios.TCGETS:
            data = struct.pack('BBBB', *info)
            jitter.vm.set_mem(arg, data)
        elif cmd == termios.TIOCGWINSZ:
            data = struct.pack('HHHH', *info)
            jitter.vm.set_mem(arg, data)
        else:
            assert data is None
        jitter.syscall_ret_systemv(0)

def sys_arml_ioctl(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (fd, cmd, arg) = jitter.syscall_args_systemv(3)
    log.debug('sys_ioctl(%x, %x, %x)', fd, cmd, arg)
    info = linux_env.ioctl(fd, cmd, arg)
    if info is False:
        jitter.syscall_ret_systemv(-1)
    else:
        if cmd == termios.TCGETS:
            data = struct.pack('BBBB', *info)
            jitter.vm.set_mem(arg, data)
        elif cmd == termios.TIOCGWINSZ:
            data = struct.pack('HHHH', *info)
            jitter.vm.set_mem(arg, data)
        else:
            assert data is None
        jitter.syscall_ret_systemv(0)

def sys_generic_open(jitter, linux_env):
    if False:
        while True:
            i = 10
    (filename, flags, mode) = jitter.syscall_args_systemv(3)
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_open(%r, %x, %x)', rpathname, flags, mode)
    jitter.syscall_ret_systemv(linux_env.open_(rpathname, flags))

def sys_generic_write(jitter, linux_env):
    if False:
        while True:
            i = 10
    (fd, buf, count) = jitter.syscall_args_systemv(3)
    log.debug('sys_write(%d, %x, %x)', fd, buf, count)
    data = jitter.vm.get_mem(buf, count)
    jitter.syscall_ret_systemv(linux_env.write(fd, data))

def sys_x86_64_getdents(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    fd = jitter.cpu.RDI
    dirent = jitter.cpu.RSI
    count = jitter.cpu.RDX
    log.debug('sys_getdents(%x, %x, %x)', fd, dirent, count)

    def packing_callback(cur_len, d_ino, d_type, name):
        if False:
            for i in range(10):
                print('nop')
        d_reclen = 8 * 2 + 2 + 1 + len(name) + 1
        d_off = cur_len + d_reclen
        entry = struct.pack('QqH', d_ino, d_off, d_reclen) + name.encode('utf8') + b'\x00' + struct.pack('B', d_type)
        assert len(entry) == d_reclen
        return entry
    out = linux_env.getdents(fd, count, packing_callback)
    jitter.vm.set_mem(dirent, out)
    jitter.cpu.RAX = len(out)

def sys_arml_getdents64(jitter, linux_env):
    if False:
        print('Hello World!')
    fd = jitter.cpu.R0
    dirent = jitter.cpu.R1
    count = jitter.cpu.R2
    log.debug('sys_getdents64(%x, %x, %x)', fd, dirent, count)

    def packing_callback(cur_len, d_ino, d_type, name):
        if False:
            i = 10
            return i + 15
        d_reclen = 8 * 2 + 2 + 1 + len(name) + 1
        d_off = cur_len + d_reclen
        entry = struct.pack('QqHB', d_ino, d_off, d_reclen, d_type) + name + b'\x00'
        assert len(entry) == d_reclen
        return entry
    out = linux_env.getdents(fd, count, packing_callback)
    jitter.vm.set_mem(dirent, out)
    jitter.cpu.R0 = len(out)

def sys_x86_64_newlstat(jitter, linux_env):
    if False:
        return 10
    filename = jitter.cpu.RDI
    statbuf = jitter.cpu.RSI
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_newlstat(%s, %x)', rpathname, statbuf)
    if not linux_env.filesystem.exists(rpathname):
        jitter.cpu.RAX = -1
    else:
        info = linux_env.lstat(rpathname)
        data = _dump_struct_stat_x86_64(info)
        jitter.vm.set_mem(statbuf, data)
        jitter.cpu.RAX = 0

def sys_arml_lstat64(jitter, linux_env):
    if False:
        while True:
            i = 10
    filename = jitter.cpu.R0
    statbuf = jitter.cpu.R1
    rpathname = jitter.get_c_str(filename)
    log.debug('sys_newlstat(%s, %x)', rpathname, statbuf)
    if not linux_env.filesystem.exists(rpathname):
        jitter.cpu.R0 = -1
    else:
        info = linux_env.lstat(rpathname)
        data = _dump_struct_stat_arml(info)
        jitter.vm.set_mem(statbuf, data)
        jitter.cpu.R0 = 0

def sys_x86_64_lgetxattr(jitter, linux_env):
    if False:
        print('Hello World!')
    pathname = jitter.cpu.RDI
    name = jitter.cpu.RSI
    value = jitter.cpu.RDX
    size = jitter.cpu.R10
    rpathname = jitter.get_c_str(pathname)
    rname = jitter.get_c_str(name)
    log.debug('sys_lgetxattr(%r, %r, %x, %x)', rpathname, rname, value, size)
    jitter.vm.set_mem(value, b'\x00' * size)
    jitter.cpu.RAX = 0

def sys_x86_64_getxattr(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    pathname = jitter.cpu.RDI
    name = jitter.cpu.RSI
    value = jitter.cpu.RDX
    size = jitter.cpu.R10
    rpathname = jitter.get_c_str(pathname)
    rname = jitter.get_c_str(name)
    log.debug('sys_getxattr(%r, %r, %x, %x)', rpathname, rname, value, size)
    jitter.vm.set_mem(value, b'\x00' * size)
    jitter.cpu.RAX = 0

def sys_x86_64_socket(jitter, linux_env):
    if False:
        print('Hello World!')
    family = jitter.cpu.RDI
    type_ = jitter.cpu.RSI
    protocol = jitter.cpu.RDX
    log.debug('sys_socket(%x, %x, %x)', family, type_, protocol)
    jitter.cpu.RAX = linux_env.socket(family, type_, protocol)

def sys_x86_64_connect(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    fd = jitter.cpu.RDI
    uservaddr = jitter.cpu.RSI
    addrlen = jitter.cpu.RDX
    raddr = jitter.get_c_str(uservaddr + 2)
    log.debug('sys_connect(%x, %r, %x)', fd, raddr, addrlen)
    jitter.cpu.RAX = -1

def sys_x86_64_clock_gettime(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    which_clock = jitter.cpu.RDI
    tp = jitter.cpu.RSI
    log.debug('sys_clock_gettime(%x, %x)', which_clock, tp)
    value = linux_env.clock_gettime()
    jitter.vm.set_mem(tp, struct.pack('Q', value))
    jitter.cpu.RAX = 0

def sys_x86_64_lseek(jitter, linux_env):
    if False:
        print('Hello World!')
    fd = jitter.cpu.RDI
    offset = jitter.cpu.RSI
    whence = jitter.cpu.RDX
    log.debug('sys_lseek(%d, %x, %x)', fd, offset, whence)
    fdesc = linux_env.file_descriptors[fd]
    mask = (1 << 64) - 1
    if offset > 1 << 63:
        offset = -((offset ^ mask) + 1)
    new_offset = fdesc.lseek(offset, whence)
    jitter.cpu.RAX = new_offset

def sys_x86_64_munmap(jitter, linux_env):
    if False:
        while True:
            i = 10
    addr = jitter.cpu.RDI
    len_ = jitter.cpu.RSI
    log.debug('sys_munmap(%x, %x)', addr, len_)
    jitter.cpu.RAX = 0

def sys_x86_64_readlink(jitter, linux_env):
    if False:
        while True:
            i = 10
    path = jitter.cpu.RDI
    buf = jitter.cpu.RSI
    bufsize = jitter.cpu.RDX
    rpath = jitter.get_c_str(path)
    log.debug('sys_readlink(%r, %x, %x)', rpath, buf, bufsize)
    link = linux_env.filesystem.readlink(rpath)
    if link is None:
        jitter.cpu.RAX = -1
    else:
        data = link[:bufsize - 1] + b'\x00'
        jitter.vm.set_mem(buf, data)
        jitter.cpu.RAX = len(data) - 1

def sys_x86_64_getpid(jitter, linux_env):
    if False:
        print('Hello World!')
    log.debug('sys_getpid()')
    jitter.cpu.RAX = linux_env.process_pid

def sys_x86_64_sysinfo(jitter, linux_env):
    if False:
        return 10
    info = jitter.cpu.RDI
    log.debug('sys_sysinfo(%x)', info)
    data = struct.pack('QQQQQQQQQQHQQI', 4660, 8192, 8192, 8192, 268435456, 268435456, 268435456, 0, 0, 0, 1, 0, 0, 1)
    jitter.vm.set_mem(info, data)
    jitter.cpu.RAX = 0

def sys_generic_geteuid(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    log.debug('sys_geteuid()')
    jitter.syscall_ret_systemv(linux_env.user_euid)

def sys_generic_getegid(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    log.debug('sys_getegid()')
    jitter.syscall_ret_systemv(linux_env.user_egid)

def sys_generic_getuid(jitter, linux_env):
    if False:
        i = 10
        return i + 15
    log.debug('sys_getuid()')
    jitter.syscall_ret_systemv(linux_env.user_uid)

def sys_generic_getgid(jitter, linux_env):
    if False:
        return 10
    log.debug('sys_getgid()')
    jitter.syscall_ret_systemv(linux_env.user_gid)

def sys_generic_setgid(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    (gid,) = jitter.syscall_args_systemv(1)
    log.debug('sys_setgid(%x)', gid)
    if gid != linux_env.user_gid:
        jitter.syscall_ret_systemv(-1)
    else:
        jitter.syscall_ret_systemv(0)

def sys_generic_setuid(jitter, linux_env):
    if False:
        return 10
    (uid,) = jitter.syscall_args_systemv(1)
    log.debug('sys_setuid(%x)', uid)
    if uid != linux_env.user_uid:
        jitter.syscall_ret_systemv(-1)
    else:
        jitter.syscall_ret_systemv(0)

def sys_arml_set_tls(jitter, linux_env):
    if False:
        for i in range(10):
            print('nop')
    ptr = jitter.cpu.R0
    log.debug('sys_set_tls(%x)', ptr)
    linux_env.tls = ptr
    jitter.cpu.R0 = 0

def sys_generic_fcntl64(jitter, linux_env):
    if False:
        return 10
    (fd, cmd, arg) = jitter.syscall_args_systemv(3)
    log.debug('sys_fcntl(%x, %x, %x)', fd, cmd, arg)
    fdesc = linux_env.file_descriptors[fd]
    if cmd == fcntl.F_GETFL:
        jitter.syscall_ret_systemv(fdesc.flags)
    elif cmd == fcntl.F_SETFL:
        jitter.syscall_ret_systemv(0)
    elif cmd == fcntl.F_GETFD:
        jitter.syscall_ret_systemv(fdesc.flags)
    elif cmd == fcntl.F_SETFD:
        jitter.syscall_ret_systemv(0)
    else:
        raise RuntimeError('Not implemented')

def sys_x86_64_pread64(jitter, linux_env):
    if False:
        print('Hello World!')
    fd = jitter.cpu.RDI
    buf = jitter.cpu.RSI
    count = jitter.cpu.RDX
    pos = jitter.cpu.R10
    log.debug('sys_pread64(%x, %x, %x, %x)', fd, buf, count, pos)
    fdesc = linux_env.file_descriptors[fd]
    cur_pos = fdesc.tell()
    fdesc.seek(pos)
    data = fdesc.read(count)
    jitter.vm.set_mem(buf, data)
    fdesc.seek(cur_pos)
    jitter.cpu.RAX = len(data)

def sys_arml_gettimeofday(jitter, linux_env):
    if False:
        return 10
    tv = jitter.cpu.R0
    tz = jitter.cpu.R1
    log.debug('sys_gettimeofday(%x, %x)', tv, tz)
    value = linux_env.clock_gettime()
    if tv:
        jitter.vm.set_mem(tv, struct.pack('II', value, 0))
    if tz:
        jitter.vm.set_mem(tz, struct.pack('II', 0, 0))
    jitter.cpu.R0 = 0

def sys_mips32b_socket(jitter, linux_env):
    if False:
        print('Hello World!')
    (family, type_, protocol) = jitter.syscall_args_systemv(3)
    log.debug('sys_socket(%x, %x, %x)', family, type_, protocol)
    ret1 = linux_env.socket(family, type_, protocol)
    jitter.syscall_ret_systemv(ret1, 0, 0)
syscall_callbacks_x86_32 = {122: sys_x86_32_newuname}
syscall_callbacks_x86_64 = {0: sys_generic_read, 1: sys_generic_write, 2: sys_generic_open, 3: sys_generic_close, 4: sys_x86_64_newstat, 5: sys_x86_64_fstat, 6: sys_x86_64_newlstat, 8: sys_x86_64_lseek, 9: sys_generic_mmap, 16: sys_x86_64_ioctl, 10: sys_generic_mprotect, 11: sys_x86_64_munmap, 12: sys_generic_brk, 13: sys_x86_64_rt_sigaction, 14: sys_x86_64_rt_sigprocmask, 17: sys_x86_64_pread64, 20: sys_x86_64_writev, 21: sys_generic_access, 39: sys_x86_64_getpid, 41: sys_x86_64_socket, 42: sys_x86_64_connect, 63: sys_x86_64_newuname, 72: sys_generic_fcntl64, 78: sys_x86_64_getdents, 89: sys_x86_64_readlink, 99: sys_x86_64_sysinfo, 102: sys_generic_getuid, 104: sys_generic_getgid, 107: sys_generic_geteuid, 108: sys_generic_getegid, 228: sys_x86_64_clock_gettime, 137: sys_x86_64_statfs, 158: sys_x86_64_arch_prctl, 191: sys_x86_64_getxattr, 192: sys_x86_64_lgetxattr, 218: sys_x86_64_set_tid_address, 231: sys_generic_exit_group, 257: sys_x86_64_openat, 273: sys_x86_64_set_robust_list, 302: sys_x86_64_prlimit64}
syscall_callbacks_arml = {3: sys_generic_read, 4: sys_generic_write, 5: sys_generic_open, 6: sys_generic_close, 45: sys_generic_brk, 33: sys_generic_access, 54: sys_arml_ioctl, 122: sys_arml_newuname, 125: sys_generic_mprotect, 146: sys_arml_writev, 192: sys_generic_mmap2, 195: sys_arml_stat64, 196: sys_arml_lstat64, 197: sys_arml_fstat64, 199: sys_generic_getuid, 200: sys_generic_getgid, 201: sys_generic_geteuid, 202: sys_generic_getegid, 78: sys_arml_gettimeofday, 213: sys_generic_setuid, 214: sys_generic_setgid, 217: sys_arml_getdents64, 221: sys_generic_fcntl64, 248: sys_generic_exit_group, 983045: sys_arml_set_tls}
syscall_callbacks_mips32b = {4183: sys_mips32b_socket}

def syscall_x86_64_exception_handler(linux_env, syscall_callbacks, jitter):
    if False:
        print('Hello World!')
    'Call to actually handle an EXCEPT_SYSCALL exception\n    In the case of an error raised by a SYSCALL, call the corresponding\n    syscall_callbacks\n    @linux_env: LinuxEnvironment_x86_64 instance\n    @syscall_callbacks: syscall number -> func(jitter, linux_env)\n    '
    syscall_number = jitter.cpu.RAX
    callback = syscall_callbacks.get(syscall_number)
    if callback is None:
        raise KeyError('No callback found for syscall number 0x%x' % syscall_number)
    callback(jitter, linux_env)
    log.debug('-> %x', jitter.cpu.RAX)
    jitter.cpu.set_exception(jitter.cpu.get_exception() ^ EXCEPT_SYSCALL)
    return True

def syscall_x86_32_exception_handler(linux_env, syscall_callbacks, jitter):
    if False:
        print('Hello World!')
    'Call to actually handle an EXCEPT_INT_XX exception\n    In the case of an error raised by a SYSCALL, call the corresponding\n    syscall_callbacks\n    @linux_env: LinuxEnvironment_x86_32 instance\n    @syscall_callbacks: syscall number -> func(jitter, linux_env)\n    '
    if jitter.cpu.interrupt_num != 128:
        return True
    syscall_number = jitter.cpu.EAX
    callback = syscall_callbacks.get(syscall_number)
    if callback is None:
        raise KeyError('No callback found for syscall number 0x%x' % syscall_number)
    callback(jitter, linux_env)
    log.debug('-> %x', jitter.cpu.EAX)
    jitter.cpu.set_exception(jitter.cpu.get_exception() ^ EXCEPT_INT_XX)
    return True

def syscall_arml_exception_handler(linux_env, syscall_callbacks, jitter):
    if False:
        print('Hello World!')
    'Call to actually handle an EXCEPT_PRIV_INSN exception\n    In the case of an error raised by a SYSCALL, call the corresponding\n    syscall_callbacks\n    @linux_env: LinuxEnvironment_arml instance\n    @syscall_callbacks: syscall number -> func(jitter, linux_env)\n    '
    if jitter.cpu.interrupt_num != 0:
        return True
    syscall_number = jitter.cpu.R7
    callback = syscall_callbacks.get(syscall_number)
    if callback is None:
        raise KeyError('No callback found for syscall number 0x%x' % syscall_number)
    callback(jitter, linux_env)
    log.debug('-> %x', jitter.cpu.R0)
    jitter.cpu.set_exception(jitter.cpu.get_exception() ^ EXCEPT_INT_XX)
    return True

def syscall_mips32b_exception_handler(linux_env, syscall_callbacks, jitter):
    if False:
        for i in range(10):
            print('nop')
    'Call to actually handle an EXCEPT_SYSCALL exception\n    In the case of an error raised by a SYSCALL, call the corresponding\n    syscall_callbacks\n    @linux_env: LinuxEnvironment_mips32b instance\n    @syscall_callbacks: syscall number -> func(jitter, linux_env)\n    '
    syscall_number = jitter.cpu.V0
    callback = syscall_callbacks.get(syscall_number)
    if callback is None:
        raise KeyError('No callback found for syscall number 0x%x' % syscall_number)
    callback(jitter, linux_env)
    log.debug('-> %x', jitter.cpu.V0)
    jitter.cpu.set_exception(jitter.cpu.get_exception() ^ EXCEPT_SYSCALL)
    return True

def enable_syscall_handling(jitter, linux_env, syscall_callbacks):
    if False:
        i = 10
        return i + 15
    'Activate handling of syscall for the current jitter instance.\n    Syscall handlers are provided by @syscall_callbacks\n    @linux_env: LinuxEnvironment instance\n    @syscall_callbacks: syscall number -> func(jitter, linux_env)\n\n    Example of use:\n    >>> linux_env = LinuxEnvironment_x86_64()\n    >>> enable_syscall_handling(jitter, linux_env, syscall_callbacks_x86_64)\n    '
    arch_name = jitter.jit.arch_name
    if arch_name == 'x8664':
        handler = syscall_x86_64_exception_handler
        handler = functools.partial(handler, linux_env, syscall_callbacks)
        jitter.add_exception_handler(EXCEPT_SYSCALL, handler)
    elif arch_name == 'x8632':
        handler = syscall_x86_32_exception_handler
        handler = functools.partial(handler, linux_env, syscall_callbacks)
        jitter.add_exception_handler(EXCEPT_INT_XX, handler)
    elif arch_name == 'arml':
        handler = syscall_arml_exception_handler
        handler = functools.partial(handler, linux_env, syscall_callbacks)
        jitter.add_exception_handler(EXCEPT_INT_XX, handler)
    elif arch_name == 'mips32b':
        handler = syscall_mips32b_exception_handler
        handler = functools.partial(handler, linux_env, syscall_callbacks)
        jitter.add_exception_handler(EXCEPT_SYSCALL, handler)
    else:
        raise ValueError('No syscall handler implemented for %s' % arch_name)