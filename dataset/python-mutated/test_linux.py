import errno
import logging
import unittest
from binascii import hexlify
import os
import shutil
import tempfile
import re
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.core.smtlib.solver import Z3Solver
from manticore.core.smtlib import BitVecVariable, issymbolic, ConstraintSet
from manticore.native import Manticore
from manticore.platforms import linux, linux_syscalls
from manticore.utils.helpers import pickle_dumps
from manticore.platforms.linux import EnvironmentError, logger as linux_logger, SymbolicFile, Linux, SLinux

class LinuxTest(unittest.TestCase):
    _multiprocess_can_split_ = True
    BIN_PATH = os.path.join(os.path.dirname(__file__), 'binaries', 'basic_linux_amd64')

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.linux = linux.Linux(self.BIN_PATH)
        self.symbolic_linux_armv7 = linux.SLinux.empty_platform('armv7')
        self.symbolic_linux_aarch64 = linux.SLinux.empty_platform('aarch64')

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        for p in [self.linux, self.symbolic_linux_armv7, self.symbolic_linux_aarch64]:
            for entry in p.fd_table.entries():
                entry.fdlike.close()

    def test_regs_init_state_x86(self) -> None:
        if False:
            i = 10
            return i + 15
        x86_defaults = {'CS': 35, 'SS': 43, 'DS': 43, 'ES': 43}
        cpu = self.linux.current
        for (reg, val) in x86_defaults.items():
            self.assertEqual(cpu.regfile.read(reg), val)

    def test_stack_init(self) -> None:
        if False:
            print('Hello World!')
        argv = ['arg1', 'arg2', 'arg3']
        real_argv = [self.BIN_PATH] + argv
        envp = ['env1', 'env2', 'env3']
        self.linux = linux.Linux(self.BIN_PATH, argv, envp)
        cpu = self.linux.current
        self.assertEqual(cpu.read_int(cpu.STACK), 4)
        argv_ptr = cpu.STACK + 8
        envp_ptr = argv_ptr + len(real_argv) * 8 + 8
        for (i, arg) in enumerate(real_argv):
            self.assertEqual(cpu.read_string(cpu.read_int(argv_ptr + i * 8)), arg)
        for (i, env) in enumerate(envp):
            self.assertEqual(cpu.read_string(cpu.read_int(envp_ptr + i * 8)), env)

    def test_symbolic_file_wildcard(self) -> None:
        if False:
            print('Hello World!')
        with tempfile.NamedTemporaryFile('w') as fp:
            fp.write('++concrete++')
            fp.flush()
            prev_log_level = linux_logger.getEffectiveLevel()
            linux_logger.setLevel(logging.DEBUG)
            with self.assertLogs(linux_logger, logging.DEBUG) as cm:
                _ = SymbolicFile(ConstraintSet(), fp.name)
            dmsg = 'Found 4 free symbolic values'
            self.assertIn(dmsg, '\n'.join(cm.output))
            with self.assertLogs(linux_logger, logging.DEBUG) as cm:
                _ = SymbolicFile(ConstraintSet(), fp.name, wildcard='+', max_size=4)
            dmsg = 'Found 4 free symbolic values'
            self.assertIn(dmsg, '\n'.join(cm.output))
            with self.assertLogs(linux_logger, logging.DEBUG) as cm:
                _ = SymbolicFile(ConstraintSet(), fp.name, wildcard='+', max_size=2)
            dmsg = 'Found 4 free symbolic values'
            wmsg = 'Found more wildcards in the file than free symbolic values allowed (4 > 2)'
            self.assertIn(wmsg, '\n'.join(cm.output))
            with self.assertLogs(linux_logger, logging.DEBUG) as cm:
                _ = SymbolicFile(ConstraintSet(), fp.name, wildcard='|')
            dmsg = 'Found 0 free symbolic values'
            self.assertIn(dmsg, '\n'.join(cm.output))
            with self.assertRaises(AssertionError) as ex:
                _ = SymbolicFile(ConstraintSet(), fp.name, wildcard='Æ')
            emsg = 'needs to be a single byte'
            self.assertIn(emsg, repr(ex.exception))
            linux_logger.setLevel(prev_log_level)

    def test_load_maps(self) -> None:
        if False:
            while True:
                i = 10
        mappings = self.linux.current.memory.mappings()
        last_map = mappings[-1]
        last_map_perms = last_map[2]
        self.assertEqual(last_map_perms, 'rwx')
        (first_map, second_map) = mappings[:2]
        first_map_name = os.path.basename(first_map[4])
        second_map_name = os.path.basename(second_map[4])
        self.assertEqual(first_map_name, 'basic_linux_amd64')
        self.assertEqual(second_map_name, 'basic_linux_amd64')

    def test_load_proc_self_maps(self) -> None:
        if False:
            i = 10
            return i + 15
        proc_maps = self.linux.current.memory.proc_self_mappings()
        maps = self.linux.current.push_bytes('/proc/self/maps\x00')
        self.assertRaises(EnvironmentError, self.linux.sys_open, maps, os.O_RDWR, None)
        self.assertRaises(EnvironmentError, self.linux.sys_open, maps, os.O_WRONLY, None)
        for i in range(1, len(proc_maps)):
            self.assertLess(proc_maps[i - 1].start, proc_maps[i].start)
            self.assertLess(proc_maps[i - 1].end, proc_maps[i].end)
        for m in proc_maps:
            self.assertNotEqual(m.start, None)
            self.assertNotEqual(m.end, None)
            self.assertNotEqual(m.perms, None)
            self.assertNotEqual(m.offset, None)
            self.assertNotEqual(m.device, None)
            self.assertNotEqual(m.inode, None)
            self.assertNotEqual(m.pathname, None)
            self.assertNotEqual(re.fullmatch('[0-9a-f]{16}\\-[0-9a-f]{16}', m.address), None)
            self.assertNotEqual(re.fullmatch('[r-][w-][x-][sp-]', m.perms), None)
            self.assertNotEqual(re.fullmatch('[0-9a-f]{16}-[0-9a-f]{16} [r-][w-][x-][sp-] [0-9a-f]{8} [0-9a-f]{2}:[0-9a-f]{2} (?=.{9})\\ *\\d+ [^\\n]*', str(m)), None)

    def test_aarch64_syscall_write(self) -> None:
        if False:
            while True:
                i = 10
        nr_write = 64
        platform = self.symbolic_linux_aarch64
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 8
        buf = platform.current.SP - 256
        s = 'hello\n'
        platform.current.write_bytes(buf, s)
        fd = 1
        size = len(s)
        platform.current.X0 = fd
        platform.current.X1 = buf
        platform.current.X2 = size
        platform.current.X8 = nr_write
        self.assertEqual(linux_syscalls.aarch64[nr_write], 'sys_write')
        platform.syscall()
        self.assertEqual(platform.current.regfile.read('X0'), size)
        res = ''.join(map(chr, platform.output.read(size)))
        self.assertEqual(res, s)

    @unittest.skip('Stat differs in different test environments')
    def test_armv7_syscall_fstat(self) -> None:
        if False:
            print('Hello World!')
        nr_fstat64 = 197
        platform = self.symbolic_linux_armv7
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 4
        filename = platform.current.push_bytes('/\x00')
        fd = platform.sys_open(filename, os.O_RDONLY, 384)
        stat = platform.current.SP - 256
        platform.current.R0 = fd
        platform.current.R1 = stat
        platform.current.R7 = nr_fstat64
        self.assertEqual(linux_syscalls.armv7[nr_fstat64], 'sys_fstat64')
        platform.syscall()
        self.assertEqual(b'02030100000000000200000000000000ed41000018000000000000000000000000000000000000000000000000000000001000000000000000100000000000000800000000000000e5c1bc5c15e85e260789ab5c8cd5db350789ab5c8cd5db3500000000', hexlify(b''.join(platform.current.read_bytes(stat, 100))))

    def test_armv7_linux_symbolic_files_workspace_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        fname = 'symfile'
        platform = self.symbolic_linux_armv7
        with open(fname, 'w') as f:
            f.write('+')
        platform.add_symbolic_file(fname)
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 4
        fname_ptr = platform.current.push_bytes(fname + '\x00')
        fd = platform.sys_open(fname_ptr, os.O_RDWR, 384)
        platform.sys_close(fd)
        files = platform.generate_workspace_files()
        os.remove(fname)
        self.assertIn(fname, files)
        self.assertEqual(len(files[fname]), 1)

    def test_armv7_linux_workspace_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        platform = self.symbolic_linux_armv7
        platform.argv = ['arg1', 'arg2']
        files = platform.generate_workspace_files()
        self.assertIn('syscalls', files)
        self.assertIn('argv', files)
        self.assertEqual(files['argv'], b'arg1\narg2\n')
        self.assertIn('env', files)
        self.assertIn('stdout', files)
        self.assertIn('stdin', files)
        self.assertIn('stderr', files)
        self.assertIn('net', files)

    def test_armv7_syscall_events(self) -> None:
        if False:
            while True:
                i = 10
        nr_fstat64 = 197

        class Receiver:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.nevents = 0

            def will_exec(self, pc, i):
                if False:
                    while True:
                        i = 10
                self.nevents += 1

            def did_exec(self, last_pc, pc, i):
                if False:
                    while True:
                        i = 10
                self.nevents += 1
        platform = self.symbolic_linux_armv7
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 4
        platform.current.memory.mmap(8192, 8192, 'rwx')
        platform.current.PC = 8192
        platform.current.write_int(platform.current.PC, 1295)
        r = Receiver()
        platform.current.subscribe('will_execute_instruction', r.will_exec)
        platform.current.subscribe('did_execute_instruction', r.did_exec)
        filename = platform.current.push_bytes('/bin/true\x00')
        fd = platform.sys_open(filename, os.O_RDONLY, 384)
        stat = platform.current.SP - 256
        platform.current.R0 = fd
        platform.current.R1 = stat
        platform.current.R7 = nr_fstat64
        self.assertEqual(linux_syscalls.armv7[nr_fstat64], 'sys_fstat64')
        pre_icount = platform.current.icount
        platform.execute()
        post_icount = platform.current.icount
        self.assertEqual(pre_icount + 1, post_icount)
        self.assertEqual(r.nevents, 2)

    def _armv7_create_openat_state(self):
        if False:
            i = 10
            return i + 15
        nr_openat = 322
        platform = self.symbolic_linux_armv7
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 4
        dir_path = tempfile.mkdtemp()
        file_name = 'file'
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'wb') as f:
            f.write(b'test')
        dirname = platform.current.push_bytes(dir_path + '\x00')
        dirfd = platform.sys_open(dirname, os.O_RDONLY, 448)
        filename = platform.current.push_bytes(file_name + '\x00')
        stat = platform.current.SP - 256
        platform.current.R0 = dirfd
        platform.current.R1 = filename
        platform.current.R2 = os.O_RDONLY
        platform.current.R3 = 448
        platform.current.R7 = nr_openat
        self.assertEqual(linux_syscalls.armv7[nr_openat], 'sys_openat')
        return (platform, dir_path)

    def test_armv7_syscall_openat_concrete(self) -> None:
        if False:
            return 10
        (platform, temp_dir) = self._armv7_create_openat_state()
        try:
            platform.syscall()
            self.assertGreater(platform.current.R0, 2)
        finally:
            shutil.rmtree(temp_dir)

    def test_armv7_syscall_openat_symbolic(self) -> None:
        if False:
            i = 10
            return i + 15
        (platform, temp_dir) = self._armv7_create_openat_state()
        try:
            platform.current.R0 = platform.constraints.new_bitvec(32, 'fd')
            with self.assertRaises(ConcretizeRegister) as cm:
                platform.syscall()
            e = cm.exception
            (_min, _max) = Z3Solver.instance().minmax(platform.constraints, e.cpu.read_register(e.reg_name))
            self.assertLess(_min, len(platform.fd_table.entries()))
            self.assertGreater(_max, len(platform.fd_table.entries()) - 1)
        finally:
            shutil.rmtree(temp_dir)

    def test_armv7_chroot(self) -> None:
        if False:
            while True:
                i = 10
        platform = self.symbolic_linux_armv7
        platform.current.memory.mmap(4096, 4096, 'rw ')
        platform.current.SP = 8192 - 4
        this_file = os.path.realpath(__file__)
        path = platform.current.push_bytes(f'{this_file}\x00')
        fd = platform.sys_chroot(path)
        self.assertEqual(fd, -errno.ENOTDIR)
        this_dir = os.path.dirname(this_file)
        path = platform.current.push_bytes(f'{this_dir}\x00')
        fd = platform.sys_chroot(path)
        self.assertEqual(fd, -errno.EPERM)

    def test_symbolic_argv_envp(self) -> None:
        if False:
            return 10
        dirname = os.path.dirname(__file__)
        self.m = Manticore.linux(os.path.join(dirname, 'binaries', 'arguments_linux_amd64'), argv=['+'], envp={'TEST': '+'})
        for state in self.m.all_states:
            ptr = state.cpu.read_int(state.cpu.RSP + 8 * 2)
            mem = state.cpu.read_bytes(ptr, 2)
            self.assertTrue(issymbolic(mem[0]))
            self.assertEqual(mem[1], b'\x00')
            ptr = state.cpu.read_int(state.cpu.RSP + 8 * 4)
            mem = state.cpu.read_bytes(ptr, 7)
            self.assertEqual(b''.join(mem[:5]), b'TEST=')
            self.assertEqual(mem[6], b'\x00')
            self.assertTrue(issymbolic(mem[5]))

    def test_serialize_state_with_closed_files(self) -> None:
        if False:
            return 10
        platform = self.linux
        filename = platform.current.push_bytes('/bin/true\x00')
        fd = platform.sys_open(filename, os.O_RDONLY, 384)
        platform.sys_close(fd)
        pickle_dumps(platform)

    def test_thumb_mode_entrypoint(self) -> None:
        if False:
            i = 10
            return i + 15
        m = Manticore.linux(os.path.join(os.path.dirname(__file__), 'binaries', 'thumb_mode_entrypoint'))
        m.context['success'] = False

        @m.init
        def init(state):
            if False:
                while True:
                    i = 10
            state.platform.current.regfile.write('R0', 0)
            state.platform.current.regfile.write('R1', 4660)
            state.platform.current.regfile.write('R2', 22136)

        @m.hook(4097)
        def pre(state):
            if False:
                while True:
                    i = 10
            state.abandon()

        @m.hook(4100)
        def post(state):
            if False:
                print('Hello World!')
            with m.locked_context() as ctx:
                ctx['success'] = state.cpu.regfile.read('R0') == 26796
            state.abandon()
        m.run()
        self.assertTrue(m.context['success'])

    def test_implemented_syscall_report(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        concrete_syscalls = set(Linux.implemented_syscalls())
        symbolic_syscalls = set(SLinux.implemented_syscalls())
        assert 'sys_read' in concrete_syscalls
        assert 'sys_read' in symbolic_syscalls
        assert 'sys_bpf' not in concrete_syscalls
        assert 'sys_tgkill' in concrete_syscalls
        assert 'sys_tgkill' not in symbolic_syscalls
        assert symbolic_syscalls.issubset(concrete_syscalls)

    def test_unimplemented_syscall_report(self) -> None:
        if False:
            print('Hello World!')
        'This test is the inverse of test_implemented_syscall_report'
        from manticore.platforms.linux_syscalls import amd64
        unimplemented_concrete_syscalls = set(Linux.unimplemented_syscalls(amd64))
        unimplemented_symbolic_syscalls = set(SLinux.unimplemented_syscalls(set(amd64.values())))
        assert 'sys_read' not in unimplemented_concrete_syscalls
        assert 'sys_read' not in unimplemented_symbolic_syscalls
        assert 'sys_bpf' in unimplemented_concrete_syscalls
        assert 'sys_tgkill' not in unimplemented_concrete_syscalls
        assert 'sys_tgkill' in unimplemented_symbolic_syscalls
        assert unimplemented_concrete_syscalls.issubset(unimplemented_symbolic_syscalls)