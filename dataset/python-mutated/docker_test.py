from __future__ import annotations
import builtins
import json
import ntpath
import os.path
import posixpath
from unittest import mock
import pytest
from pre_commit.languages import docker
from pre_commit.util import CalledProcessError
from testing.language_helpers import run_language
from testing.util import xfailif_windows
DOCKER_CGROUP_EXAMPLE = b'12:hugetlb:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n11:blkio:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n10:freezer:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n9:cpu,cpuacct:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n8:pids:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n7:rdma:/\n6:net_cls,net_prio:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n5:cpuset:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n4:devices:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n3:memory:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n2:perf_event:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n1:name=systemd:/docker/c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7\n0::/system.slice/containerd.service\n'
CONTAINER_ID = 'c33988ec7651ebc867cb24755eaf637a6734088bc7eef59d5799293a9e5450f7'
NON_DOCKER_CGROUP_EXAMPLE = b'12:perf_event:/\n11:hugetlb:/\n10:devices:/\n9:blkio:/\n8:rdma:/\n7:cpuset:/\n6:cpu,cpuacct:/\n5:freezer:/\n4:memory:/\n3:pids:/\n2:net_cls,net_prio:/\n1:name=systemd:/init.scope\n0::/init.scope\n'

def test_docker_fallback_user():
    if False:
        while True:
            i = 10

    def invalid_attribute():
        if False:
            return 10
        raise AttributeError
    with mock.patch.multiple('os', create=True, getuid=invalid_attribute, getgid=invalid_attribute):
        assert docker.get_docker_user() == ()

def test_in_docker_no_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(builtins, 'open', side_effect=FileNotFoundError):
        assert docker._is_in_docker() is False

def _mock_open(data):
    if False:
        print('Hello World!')
    return mock.patch.object(builtins, 'open', new_callable=mock.mock_open, read_data=data)

def test_in_docker_docker_in_file():
    if False:
        while True:
            i = 10
    with _mock_open(DOCKER_CGROUP_EXAMPLE):
        assert docker._is_in_docker() is True

def test_in_docker_docker_not_in_file():
    if False:
        print('Hello World!')
    with _mock_open(NON_DOCKER_CGROUP_EXAMPLE):
        assert docker._is_in_docker() is False

def test_get_container_id():
    if False:
        for i in range(10):
            print('nop')
    with _mock_open(DOCKER_CGROUP_EXAMPLE):
        assert docker._get_container_id() == CONTAINER_ID

def test_get_container_id_failure():
    if False:
        return 10
    with _mock_open(b''), pytest.raises(RuntimeError):
        docker._get_container_id()

def test_get_docker_path_not_in_docker_returns_same():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(docker, '_is_in_docker', return_value=False):
        assert docker._get_docker_path('abc') == 'abc'

@pytest.fixture
def in_docker():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(docker, '_is_in_docker', return_value=True):
        with mock.patch.object(docker, '_get_container_id', return_value=CONTAINER_ID):
            yield

def _linux_commonpath():
    if False:
        while True:
            i = 10
    return mock.patch.object(os.path, 'commonpath', posixpath.commonpath)

def _nt_commonpath():
    if False:
        print('Hello World!')
    return mock.patch.object(os.path, 'commonpath', ntpath.commonpath)

def _docker_output(out):
    if False:
        while True:
            i = 10
    ret = (0, out, b'')
    return mock.patch.object(docker, 'cmd_output_b', return_value=ret)

def test_get_docker_path_in_docker_no_binds_same_path(in_docker):
    if False:
        i = 10
        return i + 15
    docker_out = json.dumps([{'Mounts': []}]).encode()
    with _docker_output(docker_out):
        assert docker._get_docker_path('abc') == 'abc'

def test_get_docker_path_in_docker_binds_path_equal(in_docker):
    if False:
        print('Hello World!')
    binds_list = [{'Source': '/opt/my_code', 'Destination': '/project'}]
    docker_out = json.dumps([{'Mounts': binds_list}]).encode()
    with _linux_commonpath(), _docker_output(docker_out):
        assert docker._get_docker_path('/project') == '/opt/my_code'

def test_get_docker_path_in_docker_binds_path_complex(in_docker):
    if False:
        while True:
            i = 10
    binds_list = [{'Source': '/opt/my_code', 'Destination': '/project'}]
    docker_out = json.dumps([{'Mounts': binds_list}]).encode()
    with _linux_commonpath(), _docker_output(docker_out):
        path = '/project/test/something'
        assert docker._get_docker_path(path) == '/opt/my_code/test/something'

def test_get_docker_path_in_docker_no_substring(in_docker):
    if False:
        return 10
    binds_list = [{'Source': '/opt/my_code', 'Destination': '/project'}]
    docker_out = json.dumps([{'Mounts': binds_list}]).encode()
    with _linux_commonpath(), _docker_output(docker_out):
        path = '/projectSuffix/test/something'
        assert docker._get_docker_path(path) == path

def test_get_docker_path_in_docker_binds_path_many_binds(in_docker):
    if False:
        print('Hello World!')
    binds_list = [{'Source': '/something_random', 'Destination': '/not-related'}, {'Source': '/opt/my_code', 'Destination': '/project'}, {'Source': '/something-random-2', 'Destination': '/not-related-2'}]
    docker_out = json.dumps([{'Mounts': binds_list}]).encode()
    with _linux_commonpath(), _docker_output(docker_out):
        assert docker._get_docker_path('/project') == '/opt/my_code'

def test_get_docker_path_in_docker_windows(in_docker):
    if False:
        return 10
    binds_list = [{'Source': 'c:\\users\\user', 'Destination': 'c:\\folder'}]
    docker_out = json.dumps([{'Mounts': binds_list}]).encode()
    with _nt_commonpath(), _docker_output(docker_out):
        path = 'c:\\folder\\test\\something'
        expected = 'c:\\users\\user\\test\\something'
        assert docker._get_docker_path(path) == expected

def test_get_docker_path_in_docker_docker_in_docker(in_docker):
    if False:
        i = 10
        return i + 15
    err = CalledProcessError(1, (), b'', b'')
    with mock.patch.object(docker, 'cmd_output_b', side_effect=err):
        assert docker._get_docker_path('/project') == '/project'

@xfailif_windows
def test_docker_hook(tmp_path):
    if False:
        print('Hello World!')
    dockerfile = 'FROM ubuntu:22.04\nCMD ["echo", "This is overwritten by the entry"\']\n'
    tmp_path.joinpath('Dockerfile').write_text(dockerfile)
    ret = run_language(tmp_path, docker, 'echo hello hello world')
    assert ret == (0, b'hello hello world\n')