import pytest
from ...constants import *
from ...helpers import EXIT_ERROR
from ...locking import LockFailed
from ...remote import RemoteRepository
from .. import llfuse
from . import cmd, create_src_archive, RK_ENCRYPTION, read_only, fuse_mount

def test_readonly_check(archiver):
    if False:
        for i in range(10):
            print('nop')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'check', '--verify-data', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'check', '--verify-data')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'check', '--verify-data', '--bypass-lock')

def test_readonly_diff(archiver):
    if False:
        print('Hello World!')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'a')
    create_src_archive(archiver, 'b')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'diff', 'a', 'b', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'diff', 'a', 'b')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'diff', 'a', 'b', '--bypass-lock')

def test_readonly_export_tar(archiver):
    if False:
        for i in range(10):
            print('nop')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'export-tar', 'test', 'test.tar', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'export-tar', 'test', 'test.tar')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'export-tar', 'test', 'test.tar', '--bypass-lock')

def test_readonly_extract(archiver):
    if False:
        for i in range(10):
            print('nop')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'extract', 'test', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'extract', 'test')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'extract', 'test', '--bypass-lock')

def test_readonly_info(archiver):
    if False:
        print('Hello World!')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'rinfo', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'rinfo')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'rinfo', '--bypass-lock')

def test_readonly_list(archiver):
    if False:
        print('Hello World!')
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            cmd(archiver, 'rlist', exit_code=EXIT_ERROR)
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                cmd(archiver, 'rlist')
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        cmd(archiver, 'rlist', '--bypass-lock')

@pytest.mark.skipif(not llfuse, reason='llfuse not installed')
def test_readonly_mount(archiver):
    if False:
        return 10
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    create_src_archive(archiver, 'test')
    with read_only(archiver.repository_path):
        if archiver.FORK_DEFAULT:
            with fuse_mount(archiver, exit_code=EXIT_ERROR):
                pass
        else:
            with pytest.raises((LockFailed, RemoteRepository.RPCError)) as excinfo:
                with fuse_mount(archiver, fork=False):
                    pass
            if isinstance(excinfo.value, RemoteRepository.RPCError):
                assert excinfo.value.exception_class == 'LockFailed'
        with fuse_mount(archiver, None, '--bypass-lock'):
            pass