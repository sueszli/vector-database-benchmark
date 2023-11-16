import logging
import struct
import socket
import tempfile
import time
import unittest
import os
import errno
import re
from glob import glob
from pathlib import Path
from manticore.core.smtlib import Solver
from manticore.core.state import Concretize
from manticore.native import Manticore
from manticore.native.cpu.abstractcpu import ConcretizeRegister
from manticore.native.plugins import SyscallCounter
from manticore.platforms import linux, linux_syscall_stubs
from manticore.platforms.linux import SymbolicSocket, logger as linux_logger
from manticore.platforms.platform import SyscallNotImplemented, logger as platform_logger

def test_symbolic_syscall_arg() -> None:
    if False:
        while True:
            i = 10
    BIN_PATH = os.path.join(os.path.dirname(__file__), 'binaries', 'symbolic_read_count')
    tmp_dir = tempfile.TemporaryDirectory(prefix='mcore_test_')
    m = Manticore(BIN_PATH, argv=['+'], workspace_url=str(tmp_dir.name))
    m.run()
    m.finalize()
    found_win_msg = False
    win_msg = 'WIN: Read more than zero data'
    outs_glob = f'{str(m.workspace)}/test_*.stdout'
    for output_p in glob(outs_glob):
        with open(output_p) as f:
            if win_msg in f.read():
                found_win_msg = True
                break
    assert found_win_msg, f'Did not find win message in {outs_glob}: "{win_msg}"'

def test_symbolic_length_recv() -> None:
    if False:
        print('Hello World!')
    BIN_PATH = os.path.join(os.path.dirname(__file__), 'binaries', 'symbolic_length_recv')
    tmp_dir = tempfile.TemporaryDirectory(prefix='mcore_test_')
    m = Manticore(BIN_PATH, workspace_url=str(tmp_dir.name))
    m.run()
    m.finalize()
    found_msg = False
    less_len_msg = 'Received less than BUFFER_SIZE'
    outs_glob = f'{str(m.workspace)}/test_*.stdout'
    for output_p in glob(outs_glob):
        with open(output_p) as f:
            if less_len_msg in f.read():
                found_msg = True
                break
    assert found_msg, f'Did not find our message in {outs_glob}: "{less_len_msg}"'

class LinuxTest(unittest.TestCase):
    _multiprocess_can_split_ = True
    BIN_PATH = os.path.join(os.path.dirname(__file__), 'binaries', 'basic_linux_amd64')

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmp_dir = tempfile.TemporaryDirectory(prefix='mcore_test_')
        self.linux = linux.SLinux(self.BIN_PATH)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        for entry in self.linux.fd_table.entries():
            entry.fdlike.close()
        self.tmp_dir.cleanup()

    def get_path(self, basename: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns an absolute path with the given basename'
        return f'{self.tmp_dir.name}/{basename}'

    def test_time(self):
        if False:
            return 10
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        time_0 = self.linux.sys_time(0)
        self.linux.sys_clock_gettime(1, 4352)
        self.linux.sys_gettimeofday(4608, 0)
        time_2_0 = self.linux.current.read_int(4608)
        time_monotonic_0 = self.linux.current.read_int(4352)
        time.sleep(1.1)
        time_final = self.linux.sys_time(0)
        self.linux.sys_clock_gettime(1, 4352)
        self.linux.sys_gettimeofday(4608, 0)
        time_2_final = self.linux.current.read_int(4608)
        time_monotonic_final = self.linux.current.read_int(4352)
        self.assertGreater(time_monotonic_final, time_monotonic_0, 'Monotonic clock time did not increase!')
        self.assertGreater(time_final, time_0, 'Time did not increase!')
        self.assertGreater(time_2_final, time_2_0, 'Time did not increase!')

    def test_directories(self):
        if False:
            return 10
        dname = self.get_path('test_directories')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, dname)
        self.assertFalse(os.path.exists(dname))
        self.linux.sys_mkdir(4352, mode=511)
        self.assertTrue(os.path.exists(dname))
        self.linux.sys_rmdir(4352)
        self.assertFalse(os.path.exists(dname))

    def test_dir_stat(self):
        if False:
            i = 10
            return i + 15
        dname = self.get_path('test_dir_stat')
        self.assertFalse(os.path.exists(dname))
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, dname)
        self.linux.sys_mkdir(4352, mode=511)
        fd = self.linux.sys_open(4352, flags=os.O_RDONLY | os.O_DIRECTORY, mode=511)
        self.assertTrue(os.path.exists(dname))
        self.assertGreater(fd, 0)
        res = self.linux.sys_stat32(4352, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        os.rmdir(dname)
        self.assertFalse(os.path.exists(dname))
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_rmdir(4352)
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_close(fd)
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertLess(res, 0)

    def test_file_stat(self):
        if False:
            print('Hello World!')
        fname = self.get_path('test_file_stat')
        self.assertFalse(os.path.exists(fname))
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        self.assertTrue(os.path.exists(fname))
        res = self.linux.sys_stat32(4352, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        os.remove(fname)
        self.assertFalse(os.path.exists(fname))
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_unlink(4352)
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertEqual(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_close(fd)
        res = self.linux.sys_stat32(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_stat64(4352, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(-100, 4352, 4608, 0)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstatat(fd, 8191, 4608, 4096)
        self.assertLess(res, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertLess(res, 0)

    def test_socketdesc_stat(self):
        if False:
            print('Hello World!')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        fd = self.linux.sys_socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_close(fd)
        res = self.linux.sys_newfstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat(fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat64(fd, 4608)
        self.assertLess(res, 0)

    def test_socket_stat(self):
        if False:
            i = 10
            return i + 15
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        sock_fd = self.linux.sys_socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.linux.sys_bind(sock_fd, None, None)
        self.linux.sys_listen(sock_fd, None)
        conn_fd = self.linux.sys_accept(sock_fd, None, 0)
        res = self.linux.sys_newfstat(conn_fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat(conn_fd, 4608)
        self.assertEqual(res, 0)
        res = self.linux.sys_fstat64(conn_fd, 4608)
        self.assertEqual(res, 0)
        self.linux.sys_close(conn_fd)
        res = self.linux.sys_newfstat(conn_fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat(conn_fd, 4608)
        self.assertLess(res, 0)
        res = self.linux.sys_fstat64(conn_fd, 4608)
        self.assertLess(res, 0)

    def test_pipe(self):
        if False:
            while True:
                i = 10
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.sys_pipe(4352)
        fd1 = self.linux.current.read_int(4352, 8 * 4)
        fd2 = self.linux.current.read_int(4352 + 4, 8 * 4)
        buf = b'0123456789ABCDEF'
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd1, 4608, len(buf))
        self.linux.sys_read(fd2, 4864, len(buf))
        self.assertEqual(buf, b''.join(self.linux.current.read_bytes(4864, len(buf))), 'Pipe Read/Write failed')

    def test_ftruncate(self):
        if False:
            while True:
                i = 10
        fname = self.get_path('test_ftruncate')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'0123456789ABCDEF'
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        self.linux.sys_close(fd)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        self.linux.sys_ftruncate(fd, len(buf) // 2)
        self.linux.sys_read(fd, 4864, len(buf))
        self.assertEqual(buf[:8] + b'\x00' * 8, b''.join(self.linux.current.read_bytes(4864, len(buf))))

    def test_link(self):
        if False:
            i = 10
            return i + 15
        fname = self.get_path('test_link_from')
        newname = self.get_path('test_link_to')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        self.linux.current.write_string(4480, newname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'0123456789ABCDEF'
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        self.linux.sys_close(fd)
        self.linux.sys_link(4352, 4480)
        self.assertTrue(os.path.exists(newname))
        fd = self.linux.sys_open(4480, os.O_RDWR, 511)
        self.linux.sys_read(fd, 4864, len(buf))
        self.assertEqual(buf, b''.join(self.linux.current.read_bytes(4864, len(buf))))
        self.linux.sys_close(fd)
        self.linux.sys_unlink(4480)
        self.assertFalse(os.path.exists(newname))

    def test_chmod(self):
        if False:
            while True:
                i = 10
        fname = self.get_path('test_chmod')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        print('Creating', fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'0123456789ABCDEF'
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_close(fd)
        self.linux.sys_chmod(4352, 292)
        self.assertEqual(-errno.EACCES, self.linux.sys_open(4352, os.O_WRONLY, 511))
        self.assertEqual(-errno.EPERM, self.linux.sys_chown(4352, 0, 0))

    def test_read_symb_socket(self):
        if False:
            return 10
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        sock_fd = self.linux.sys_socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.assertEqual(sock_fd, 3)
        self.linux.sys_bind(sock_fd, None, None)
        self.linux.sys_listen(sock_fd, None)
        conn_fd = self.linux.sys_accept(sock_fd, None, 0)
        self.assertEqual(conn_fd, 4)
        sock_obj = self.linux.fd_table.get_fdlike(conn_fd)
        assert isinstance(sock_obj, SymbolicSocket)
        init_len = len(sock_obj.buffer)
        self.assertEqual(init_len, 0)
        BYTES = 5
        sock_obj._symb_len = BYTES
        wrote = self.linux.sys_read(conn_fd, 4352, BYTES)
        self.assertEqual(wrote, BYTES)
        BYTES = 100
        sock_obj._symb_len = BYTES
        wrote = self.linux.sys_read(conn_fd, 0, BYTES)
        self.assertEqual(wrote, -errno.EFAULT)
        remaining_bytes = sock_obj.max_recv_symbolic - sock_obj.recv_pos
        BYTES = remaining_bytes + 10
        sock_obj._symb_len = remaining_bytes
        wrote = self.linux.sys_read(conn_fd, 4352, BYTES)
        self.assertNotEqual(wrote, BYTES)
        self.assertEqual(wrote, remaining_bytes)
        BYTES = 10
        sock_obj._symb_len = 0
        wrote = self.linux.sys_read(conn_fd, 4352, BYTES)
        self.assertEqual(wrote, 0)
        BYTES = 10
        sock_obj._symb_len = BYTES
        self.linux.sys_close(conn_fd)
        wrote = self.linux.sys_read(conn_fd, 4352, BYTES)
        self.assertEqual(wrote, -errno.EBADF)

    def test_recvfrom_symb_socket(self):
        if False:
            for i in range(10):
                print('nop')
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        sock_fd = self.linux.sys_socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.assertEqual(sock_fd, 3)
        self.linux.sys_bind(sock_fd, None, None)
        self.linux.sys_listen(sock_fd, None)
        conn_fd = self.linux.sys_accept(sock_fd, None, 0)
        self.assertEqual(conn_fd, 4)
        sock_obj = self.linux.fd_table.get_fdlike(conn_fd)
        assert isinstance(sock_obj, SymbolicSocket)
        init_len = len(sock_obj.buffer)
        self.assertEqual(init_len, 0)
        BYTES = 5
        sock_obj._symb_len = BYTES
        wrote = self.linux.sys_recvfrom(conn_fd, 4352, BYTES, 0, 0, 0)
        self.assertEqual(wrote, BYTES)
        wrote = self.linux.sys_recvfrom(conn_fd, 0, 100, 0, 0, 0)
        self.assertEqual(wrote, -errno.EFAULT)
        remaining_bytes = sock_obj.max_recv_symbolic - sock_obj.recv_pos
        BYTES = remaining_bytes + 10
        sock_obj._symb_len = remaining_bytes
        wrote = self.linux.sys_recvfrom(conn_fd, 4352, BYTES, 0, 0, 0)
        self.assertNotEqual(wrote, BYTES)
        self.assertEqual(wrote, remaining_bytes)
        BYTES = 10
        sock_obj._symb_len = 0
        wrote = self.linux.sys_recvfrom(conn_fd, 4352, BYTES, 0, 0, 0)
        self.assertEqual(wrote, 0)
        self.linux.sys_close(conn_fd)
        BYTES = 10
        sock_obj._symb_len = 0
        wrote = self.linux.sys_recvfrom(conn_fd, 4352, BYTES, 0, 0, 0)
        self.assertEqual(wrote, -errno.EBADF)

    def test_multiple_sockets(self):
        if False:
            while True:
                i = 10
        sock_fd = self.linux.sys_socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self.assertEqual(sock_fd, 3)
        self.linux.sys_bind(sock_fd, None, None)
        self.linux.sys_listen(sock_fd, None)
        conn_fd = self.linux.sys_accept(sock_fd, None, 0)
        self.assertEqual(conn_fd, 4)
        self.linux.sys_close(conn_fd)
        conn_fd = -1
        conn_fd = self.linux.sys_accept(sock_fd, None, 0)
        self.assertEqual(conn_fd, 4)

    def test_lseek(self):
        if False:
            print('Hello World!')
        fname = self.get_path('test_lseek')
        assert len(fname) < 256
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'1' * 1000
        self.assertEqual(len(buf), 1000)
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        pos = self.linux.sys_lseek(fd, 100, os.SEEK_SET)
        self.assertEqual(100, pos)
        pos = self.linux.sys_lseek(fd, -50, os.SEEK_CUR)
        self.assertEqual(50, pos)
        pos = self.linux.sys_lseek(fd, 50, os.SEEK_CUR)
        self.assertEqual(100, pos)
        pos = self.linux.sys_lseek(fd, 0, os.SEEK_END)
        self.assertEqual(len(buf), pos)
        pos = self.linux.sys_lseek(fd, -50, os.SEEK_END)
        self.assertEqual(len(buf) - 50, pos)
        pos = self.linux.sys_lseek(fd, 50, os.SEEK_END)
        self.assertEqual(len(buf) + 50, pos)
        self.linux.sys_close(fd)
        pos = self.linux.sys_lseek(fd, 0, os.SEEK_SET)
        self.assertEqual(-errno.EBADF, pos)

    @unittest.expectedFailure
    def test_lseek_end_broken(self):
        if False:
            print('Hello World!')
        fname = self.get_path('test_lseek')
        assert len(fname) < 256
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'1' * 1000
        self.assertEqual(len(buf), 1000)
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        pos = self.linux.sys_lseek(fd, -2 * len(buf), os.SEEK_END)
        self.assertEqual(-errno.EBADF, pos)

    def test_llseek(self):
        if False:
            return 10
        fname = self.get_path('test_llseek')
        assert len(fname) < 256
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'1' * 1000
        self.assertEqual(len(buf), 1000)
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        result_struct = struct.Struct('q')
        resultp = 6400
        result_size = result_struct.size

        def read_resultp():
            if False:
                for i in range(10):
                    print('nop')
            'reads the `loff_t` value -- a long long -- from the result pointer'
            data = self.linux.current.read_bytes(resultp, result_struct.size)
            return result_struct.unpack(b''.join(data))[0]
        res = self.linux.sys_llseek(fd, 0, 100, resultp, os.SEEK_SET)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), 100)
        res = self.linux.sys_llseek(fd, 1, 0, resultp, os.SEEK_CUR)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), 4294967396)
        res = self.linux.sys_llseek(fd, 0, -1000, resultp, os.SEEK_CUR)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), 4294966396)
        res = self.linux.sys_llseek(fd, 0, 0, resultp, os.SEEK_END)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), len(buf))
        res = self.linux.sys_llseek(fd, 0, 50, resultp, os.SEEK_END)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), len(buf) + 50)
        res = self.linux.sys_llseek(fd, 0, -50, resultp, os.SEEK_END)
        self.assertEqual(res, 0)
        self.assertEqual(read_resultp(), len(buf) - 50)
        self.linux.sys_close(fd)
        res = self.linux.sys_llseek(fd, 0, 0, resultp, os.SEEK_SET)
        self.assertEqual(-errno.EBADF, res)

    @unittest.expectedFailure
    def test_llseek_end_broken(self):
        if False:
            print('Hello World!')
        fname = self.get_path('test_llseek_end_broken')
        assert len(fname) < 256
        self.linux.current.memory.mmap(4096, 4096, 'rw')
        self.linux.current.write_string(4352, fname)
        fd = self.linux.sys_open(4352, os.O_RDWR, 511)
        buf = b'1' * 1000
        self.assertEqual(len(buf), 1000)
        self.linux.current.write_bytes(4608, buf)
        self.linux.sys_write(fd, 4608, len(buf))
        resultp = 6400
        res = self.linux.sys_llseek(fd, 0, -2 * len(buf), resultp, os.SEEK_END)
        self.assertTrue(res < 0)

    class test_epoll(unittest.TestCase):

        def test_fork_unique_solution(self):
            if False:
                for i in range(10):
                    print('nop')
            binary = str(Path(__file__).parent.parent.parent.joinpath('tests', 'native', 'binaries', 'epoll'))
            tmp_dir = tempfile.TemporaryDirectory(prefix='mcore_test_epoll')
            m = Manticore(binary, stdin_size=5, workspace_url=str(tmp_dir.name), concrete_start='stop\n')
            counter = SyscallCounter()
            m.register_plugin(counter)
            m.run()
            m.finalize()
            syscall_counts = counter.get_counts()
            self.assertEqual(syscall_counts['sys_epoll_create1'], 1)
            self.assertEqual(syscall_counts['sys_epoll_ctl'], 1)
            self.assertEqual(syscall_counts['sys_epoll_wait'], 1)

    def test_unimplemented_symbolic_syscall(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cpu = self.linux.current
        cpu.RDI = self.linux.constraints.new_bitvec(cpu.address_bit_size, 'addr')
        cpu.RAX = 12
        prev_log_level = linux_logger.getEffectiveLevel()
        linux_logger.setLevel(logging.DEBUG)
        with self.assertLogs(linux_logger, logging.DEBUG) as cm:
            with self.assertRaises(ConcretizeRegister):
                self.linux.syscall()
        dmsg = 'Unimplemented symbolic argument to sys_brk. Concretizing argument 0'
        self.assertIn(dmsg, '\n'.join(cm.output))
        linux_logger.setLevel(prev_log_level)

    def test_unimplemented_stubs(self) -> None:
        if False:
            while True:
                i = 10
        stubs = linux_syscall_stubs.SyscallStubs(default_to_fail=False)
        with self.assertLogs(platform_logger, logging.WARNING) as cm:
            self.assertRaises(SyscallNotImplemented, stubs.sys_bpf, 0, 0, 0)
        pat = re.compile('Unimplemented system call: .+: .+\\(.+\\)', re.MULTILINE)
        self.assertRegex('\n'.join(cm.output), pat)
        self.linux.stubs.default_to_fail = False
        self.linux.current.RAX = 321
        self.assertRaises(SyscallNotImplemented, self.linux.syscall)
        self.linux.stubs.default_to_fail = True
        self.linux.current.RAX = 321
        self.linux.syscall()
        self.assertEqual(18446744073709551615, self.linux.current.RAX)

    def test_unimplemented_linux(self) -> None:
        if False:
            return 10
        with self.assertLogs(platform_logger, logging.WARNING) as cm:
            self.linux.sys_futex(0, 0, 0, 0, 0, 0)
        pat = re.compile('Unimplemented system call: .+: .+\\(.+\\)', re.MULTILINE)
        self.assertRegex('\n'.join(cm.output), pat)