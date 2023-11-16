import datetime
import json
import signal
import docker
from docker.api import APIClient
from unittest import mock
import pytest
from . import fake_api
from ..helpers import requires_api_version
from .api_test import BaseAPIClientTest, url_prefix, fake_request, DEFAULT_TIMEOUT_SECONDS, fake_inspect_container, url_base

def fake_inspect_container_tty(self, container):
    if False:
        while True:
            i = 10
    return fake_inspect_container(self, container, tty=True)

class StartContainerTest(BaseAPIClientTest):

    def test_start_container(self):
        if False:
            i = 10
            return i + 15
        self.client.start(fake_api.FAKE_CONTAINER_ID)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/start'
        assert 'data' not in args[1]
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_start_container_none(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError) as excinfo:
            self.client.start(container=None)
        assert str(excinfo.value) == 'Resource ID was not provided'
        with pytest.raises(ValueError) as excinfo:
            self.client.start(None)
        assert str(excinfo.value) == 'Resource ID was not provided'

    def test_start_container_regression_573(self):
        if False:
            i = 10
            return i + 15
        self.client.start(**{'container': fake_api.FAKE_CONTAINER_ID})

    def test_start_container_with_lxc_conf(self):
        if False:
            while True:
                i = 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, lxc_conf={'lxc.conf.k': 'lxc.conf.value'})

    def test_start_container_with_lxc_conf_compat(self):
        if False:
            return 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, lxc_conf=[{'Key': 'lxc.conf.k', 'Value': 'lxc.conf.value'}])

    def test_start_container_with_binds_ro(self):
        if False:
            return 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, binds={'/tmp': {'bind': '/mnt', 'ro': True}})

    def test_start_container_with_binds_rw(self):
        if False:
            return 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, binds={'/tmp': {'bind': '/mnt', 'ro': False}})

    def test_start_container_with_port_binds(self):
        if False:
            while True:
                i = 10
        self.maxDiff = None
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, port_bindings={1111: None, 2222: 2222, '3333/udp': (3333,), 4444: ('127.0.0.1',), 5555: ('127.0.0.1', 5555), 6666: [('127.0.0.1',), ('192.168.0.1',)]})

    def test_start_container_with_links(self):
        if False:
            while True:
                i = 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, links={'path': 'alias'})

    def test_start_container_with_multiple_links(self):
        if False:
            return 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, links={'path1': 'alias1', 'path2': 'alias2'})

    def test_start_container_with_links_as_list_of_tuples(self):
        if False:
            while True:
                i = 10
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, links=[('path', 'alias')])

    def test_start_container_privileged(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(docker.errors.DeprecatedMethod):
            self.client.start(fake_api.FAKE_CONTAINER_ID, privileged=True)

    def test_start_container_with_dict_instead_of_id(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.start({'Id': fake_api.FAKE_CONTAINER_ID})
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/start'
        assert 'data' not in args[1]
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

class CreateContainerTest(BaseAPIClientTest):

    def test_create_container(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'true')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox", "Cmd": ["true"],\n             "AttachStdin": false,\n             "AttachStderr": true, "AttachStdout": true,\n             "StdinOnce": false,\n             "OpenStdin": false, "NetworkDisabled": false}\n         ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_binds(self):
        if False:
            return 10
        mount_dest = '/mnt'
        self.client.create_container('busybox', ['ls', mount_dest], volumes=[mount_dest])
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls", "/mnt"], "AttachStdin": false,\n             "Volumes": {"/mnt": {}},\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_volume_string(self):
        if False:
            i = 10
            return i + 15
        mount_dest = '/mnt'
        self.client.create_container('busybox', ['ls', mount_dest], volumes=mount_dest)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls", "/mnt"], "AttachStdin": false,\n             "Volumes": {"/mnt": {}},\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_ports(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'ls', ports=[1111, (2222, 'udp'), (3333,)])
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "ExposedPorts": {\n                "1111/tcp": {},\n                "2222/udp": {},\n                "3333/tcp": {}\n             },\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_entrypoint(self):
        if False:
            return 10
        self.client.create_container('busybox', 'hello', entrypoint='cowsay entry')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["hello"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "Entrypoint": ["cowsay", "entry"]}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_host_config_cpu_shares(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(cpu_shares=512))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "HostConfig": {\n                "CpuShares": 512,\n                "NetworkMode": "default"\n             }}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_host_config_cpuset(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(cpuset_cpus='0,1'))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "HostConfig": {\n                "CpusetCpus": "0,1",\n                "NetworkMode": "default"\n             }}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_host_config_cpuset_mems(self):
        if False:
            print('Hello World!')
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(cpuset_mems='0'))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "HostConfig": {\n                "CpusetMems": "0",\n                "NetworkMode": "default"\n            }}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_cgroup_parent(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(cgroup_parent='test'))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        data = json.loads(args[1]['data'])
        assert 'HostConfig' in data
        assert 'CgroupParent' in data['HostConfig']
        assert data['HostConfig']['CgroupParent'] == 'test'

    def test_create_container_with_working_dir(self):
        if False:
            return 10
        self.client.create_container('busybox', 'ls', working_dir='/root')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "WorkingDir": "/root"}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_stdin_open(self):
        if False:
            return 10
        self.client.create_container('busybox', 'true', stdin_open=True)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox", "Cmd": ["true"],\n             "AttachStdin": true,\n             "AttachStderr": true, "AttachStdout": true,\n             "StdinOnce": true,\n             "OpenStdin": true, "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_named_container(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.create_container('busybox', 'true', name='marisa-kirisame')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox", "Cmd": ["true"],\n             "AttachStdin": false,\n             "AttachStderr": true, "AttachStdout": true,\n             "StdinOnce": false,\n             "OpenStdin": false, "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['params'] == {'name': 'marisa-kirisame'}

    def test_create_container_with_platform(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'true', platform='linux')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox", "Cmd": ["true"],\n             "AttachStdin": false,\n             "AttachStderr": true, "AttachStdout": true,\n             "StdinOnce": false,\n             "OpenStdin": false, "NetworkDisabled": false}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['params'] == {'name': None, 'platform': 'linux'}

    def test_create_container_with_mem_limit_as_int(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(mem_limit=128.0))
        args = fake_request.call_args
        data = json.loads(args[1]['data'])
        assert data['HostConfig']['Memory'] == 128.0

    def test_create_container_with_mem_limit_as_string(self):
        if False:
            return 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(mem_limit='128'))
        args = fake_request.call_args
        data = json.loads(args[1]['data'])
        assert data['HostConfig']['Memory'] == 128.0

    def test_create_container_with_mem_limit_as_string_with_k_unit(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(mem_limit='128k'))
        args = fake_request.call_args
        data = json.loads(args[1]['data'])
        assert data['HostConfig']['Memory'] == 128.0 * 1024

    def test_create_container_with_mem_limit_as_string_with_m_unit(self):
        if False:
            print('Hello World!')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(mem_limit='128m'))
        args = fake_request.call_args
        data = json.loads(args[1]['data'])
        assert data['HostConfig']['Memory'] == 128.0 * 1024 * 1024

    def test_create_container_with_mem_limit_as_string_with_g_unit(self):
        if False:
            print('Hello World!')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(mem_limit='128g'))
        args = fake_request.call_args
        data = json.loads(args[1]['data'])
        assert data['HostConfig']['Memory'] == 128.0 * 1024 * 1024 * 1024

    def test_create_container_with_mem_limit_as_string_with_wrong_value(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(docker.errors.DockerException):
            self.client.create_host_config(mem_limit='128p')
        with pytest.raises(docker.errors.DockerException):
            self.client.create_host_config(mem_limit='1f28')

    def test_create_container_with_lxc_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(lxc_conf={'lxc.conf.k': 'lxc.conf.value'}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['LxcConf'] = [{'Value': 'lxc.conf.value', 'Key': 'lxc.conf.k'}]
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_lxc_conf_compat(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(lxc_conf=[{'Key': 'lxc.conf.k', 'Value': 'lxc.conf.value'}]))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['LxcConf'] = [{'Value': 'lxc.conf.value', 'Key': 'lxc.conf.k'}]
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_binds_ro(self):
        if False:
            return 10
        mount_dest = '/mnt'
        mount_origin = '/tmp'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(binds={mount_origin: {'bind': mount_dest, 'ro': True}}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Binds'] = ['/tmp:/mnt:ro']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_binds_rw(self):
        if False:
            i = 10
            return i + 15
        mount_dest = '/mnt'
        mount_origin = '/tmp'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(binds={mount_origin: {'bind': mount_dest, 'ro': False}}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Binds'] = ['/tmp:/mnt:rw']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_binds_mode(self):
        if False:
            print('Hello World!')
        mount_dest = '/mnt'
        mount_origin = '/tmp'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(binds={mount_origin: {'bind': mount_dest, 'mode': 'z'}}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Binds'] = ['/tmp:/mnt:z']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_binds_mode_and_ro_error(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError):
            mount_dest = '/mnt'
            mount_origin = '/tmp'
            self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(binds={mount_origin: {'bind': mount_dest, 'mode': 'z', 'ro': True}}))

    def test_create_container_with_binds_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(binds=['/tmp:/mnt/1:ro', '/tmp:/mnt/2']))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Binds'] = ['/tmp:/mnt/1:ro', '/tmp:/mnt/2']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_port_binds(self):
        if False:
            print('Hello World!')
        self.maxDiff = None
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(port_bindings={1111: None, 2222: 2222, '3333/udp': (3333,), 4444: ('127.0.0.1',), 5555: ('127.0.0.1', 5555), 6666: [('127.0.0.1',), ('192.168.0.1',)]}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        data = json.loads(args[1]['data'])
        port_bindings = data['HostConfig']['PortBindings']
        assert '1111/tcp' in port_bindings
        assert '2222/tcp' in port_bindings
        assert '3333/udp' in port_bindings
        assert '4444/tcp' in port_bindings
        assert '5555/tcp' in port_bindings
        assert '6666/tcp' in port_bindings
        assert [{'HostPort': '', 'HostIp': ''}] == port_bindings['1111/tcp']
        assert [{'HostPort': '2222', 'HostIp': ''}] == port_bindings['2222/tcp']
        assert [{'HostPort': '3333', 'HostIp': ''}] == port_bindings['3333/udp']
        assert [{'HostPort': '', 'HostIp': '127.0.0.1'}] == port_bindings['4444/tcp']
        assert [{'HostPort': '5555', 'HostIp': '127.0.0.1'}] == port_bindings['5555/tcp']
        assert len(port_bindings['6666/tcp']) == 2
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_mac_address(self):
        if False:
            return 10
        expected = '02:42:ac:11:00:0a'
        self.client.create_container('busybox', ['sleep', '60'], mac_address=expected)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        data = json.loads(args[1]['data'])
        assert data['MacAddress'] == expected

    def test_create_container_with_links(self):
        if False:
            print('Hello World!')
        link_path = 'path'
        alias = 'alias'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(links={link_path: alias}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Links'] = ['path:alias']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_multiple_links(self):
        if False:
            print('Hello World!')
        link_path = 'path'
        alias = 'alias'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(links={link_path + '1': alias + '1', link_path + '2': alias + '2'}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Links'] = ['path1:alias1', 'path2:alias2']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_with_links_as_list_of_tuples(self):
        if False:
            for i in range(10):
                print('nop')
        link_path = 'path'
        alias = 'alias'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(links=[(link_path, alias)]))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Links'] = ['path:alias']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    def test_create_container_privileged(self):
        if False:
            return 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(privileged=True))
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Privileged'] = True
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_restart_policy(self):
        if False:
            while True:
                i = 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(restart_policy={'Name': 'always', 'MaximumRetryCount': 0}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['RestartPolicy'] = {'MaximumRetryCount': 0, 'Name': 'always'}
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_added_capabilities(self):
        if False:
            print('Hello World!')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(cap_add=['MKNOD']))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['CapAdd'] = ['MKNOD']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_dropped_capabilities(self):
        if False:
            while True:
                i = 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(cap_drop=['MKNOD']))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['CapDrop'] = ['MKNOD']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_devices(self):
        if False:
            while True:
                i = 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(devices=['/dev/sda:/dev/xvda:rwm', '/dev/sdb:/dev/xvdb', '/dev/sdc']))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Devices'] = [{'CgroupPermissions': 'rwm', 'PathInContainer': '/dev/xvda', 'PathOnHost': '/dev/sda'}, {'CgroupPermissions': 'rwm', 'PathInContainer': '/dev/xvdb', 'PathOnHost': '/dev/sdb'}, {'CgroupPermissions': 'rwm', 'PathInContainer': '/dev/sdc', 'PathOnHost': '/dev/sdc'}]
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_device_requests(self):
        if False:
            for i in range(10):
                print('nop')
        client = APIClient(version='1.40')
        fake_api.fake_responses.setdefault(f'{fake_api.prefix}/v1.40/containers/create', fake_api.post_fake_create_container)
        client.create_container('busybox', 'true', host_config=client.create_host_config(device_requests=[{'device_ids': ['0', 'GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a']}, {'driver': 'nvidia', 'Count': -1, 'capabilities': [['gpu', 'utility']], 'options': {'key': 'value'}}]))
        args = fake_request.call_args
        assert args[0][1] == url_base + 'v1.40/' + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = client.create_host_config()
        expected_payload['HostConfig']['DeviceRequests'] = [{'Driver': '', 'Count': 0, 'DeviceIDs': ['0', 'GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a'], 'Capabilities': [], 'Options': {}}, {'Driver': 'nvidia', 'Count': -1, 'DeviceIDs': [], 'Capabilities': [['gpu', 'utility']], 'Options': {'key': 'value'}}]
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers']['Content-Type'] == 'application/json'
        assert set(args[1]['headers']) <= {'Content-Type', 'User-Agent'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_labels_dict(self):
        if False:
            i = 10
            return i + 15
        labels_dict = {'foo': '1', 'bar': '2'}
        self.client.create_container('busybox', 'true', labels=labels_dict)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data'])['Labels'] == labels_dict
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_labels_list(self):
        if False:
            while True:
                i = 10
        labels_list = ['foo', 'bar']
        labels_dict = {'foo': '', 'bar': ''}
        self.client.create_container('busybox', 'true', labels=labels_list)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data'])['Labels'] == labels_dict
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_named_volume(self):
        if False:
            print('Hello World!')
        mount_dest = '/mnt'
        volume_name = 'name'
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(volume_driver='foodriver', binds={volume_name: {'bind': mount_dest, 'ro': False}}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['VolumeDriver'] = 'foodriver'
        expected_payload['HostConfig']['Binds'] = ['name:/mnt:rw']
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_stop_signal(self):
        if False:
            while True:
                i = 10
        self.client.create_container('busybox', 'ls', stop_signal='SIGINT')
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "StopSignal": "SIGINT"}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    @requires_api_version('1.22')
    def test_create_container_with_aliases(self):
        if False:
            return 10
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(network_mode='some-network'), networking_config=self.client.create_networking_config({'some-network': self.client.create_endpoint_config(aliases=['foo', 'bar'])}))
        args = fake_request.call_args
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "HostConfig": {\n               "NetworkMode": "some-network"\n             },\n             "NetworkingConfig": {\n               "EndpointsConfig": {\n                 "some-network": {"Aliases": ["foo", "bar"]}\n               }\n            }}\n        ')

    @requires_api_version('1.22')
    def test_create_container_with_tmpfs_list(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(tmpfs=['/tmp', '/mnt:size=3G,uid=100']))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Tmpfs'] = {'/tmp': '', '/mnt': 'size=3G,uid=100'}
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    @requires_api_version('1.22')
    def test_create_container_with_tmpfs_dict(self):
        if False:
            return 10
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(tmpfs={'/tmp': '', '/mnt': 'size=3G,uid=100'}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Tmpfs'] = {'/tmp': '', '/mnt': 'size=3G,uid=100'}
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    @requires_api_version('1.24')
    def test_create_container_with_sysctl(self):
        if False:
            print('Hello World!')
        self.client.create_container('busybox', 'true', host_config=self.client.create_host_config(sysctls={'net.core.somaxconn': 1024, 'net.ipv4.tcp_syncookies': '0'}))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['Sysctls'] = {'net.core.somaxconn': '1024', 'net.ipv4.tcp_syncookies': '0'}
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}
        assert args[1]['timeout'] == DEFAULT_TIMEOUT_SECONDS

    def test_create_container_with_unicode_envvars(self):
        if False:
            print('Hello World!')
        envvars_dict = {'foo': '☃'}
        expected = ['foo=☃']
        self.client.create_container('busybox', 'true', environment=envvars_dict)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data'])['Env'] == expected

    @requires_api_version('1.25')
    def test_create_container_with_host_config_cpus(self):
        if False:
            i = 10
            return i + 15
        self.client.create_container('busybox', 'ls', host_config=self.client.create_host_config(cpu_count=1, cpu_percent=20, nano_cpus=1000))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        assert json.loads(args[1]['data']) == json.loads('\n            {"Tty": false, "Image": "busybox",\n             "Cmd": ["ls"], "AttachStdin": false,\n             "AttachStderr": true,\n             "AttachStdout": true, "OpenStdin": false,\n             "StdinOnce": false,\n             "NetworkDisabled": false,\n             "HostConfig": {\n                "CpuCount": 1,\n                "CpuPercent": 20,\n                "NanoCpus": 1000,\n                "NetworkMode": "default"\n            }}\n        ')
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

    @requires_api_version('1.41')
    def test_create_container_with_cgroupns(self):
        if False:
            print('Hello World!')
        self.client.create_container(image='busybox', command='true', host_config=self.client.create_host_config(cgroupns='private'))
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/create'
        expected_payload = self.base_create_payload()
        expected_payload['HostConfig'] = self.client.create_host_config()
        expected_payload['HostConfig']['CgroupnsMode'] = 'private'
        assert json.loads(args[1]['data']) == expected_payload
        assert args[1]['headers'] == {'Content-Type': 'application/json'}

class ContainerTest(BaseAPIClientTest):

    def test_list_containers(self):
        if False:
            while True:
                i = 10
        self.client.containers(all=True)
        fake_request.assert_called_with('GET', url_prefix + 'containers/json', params={'all': 1, 'since': None, 'size': 0, 'limit': -1, 'trunc_cmd': 0, 'before': None}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_resize_container(self):
        if False:
            i = 10
            return i + 15
        self.client.resize({'Id': fake_api.FAKE_CONTAINER_ID}, height=15, width=120)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/resize', params={'h': 15, 'w': 120}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_rename_container(self):
        if False:
            while True:
                i = 10
        self.client.rename({'Id': fake_api.FAKE_CONTAINER_ID}, name='foobar')
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/rename', params={'name': 'foobar'}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_wait(self):
        if False:
            i = 10
            return i + 15
        self.client.wait(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/wait', timeout=None, params={})

    def test_wait_with_dict_instead_of_id(self):
        if False:
            print('Hello World!')
        self.client.wait({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/wait', timeout=None, params={})

    def test_logs(self):
        if False:
            return 10
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            logs = self.client.logs(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)
        assert logs == b'Flowering Nights\n(Sakuya Iyazoi)\n'

    def test_logs_with_dict_instead_of_id(self):
        if False:
            return 10
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            logs = self.client.logs({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)
        assert logs == b'Flowering Nights\n(Sakuya Iyazoi)\n'

    def test_log_streaming(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=True, follow=False)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=True)

    def test_log_following(self):
        if False:
            print('Hello World!')
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=True)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 1, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)

    def test_log_following_backwards(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=True)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 1, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=True)

    def test_log_streaming_and_following(self):
        if False:
            return 10
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=True, follow=True)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 1, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=True)

    def test_log_tail(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=False, tail=10)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 10}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)

    def test_log_since(self):
        if False:
            for i in range(10):
                print('nop')
        ts = 809222400
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=False, since=ts)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all', 'since': ts}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)

    def test_log_since_with_float(self):
        if False:
            i = 10
            return i + 15
        ts = 809222400.0
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=False, since=ts)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all', 'since': ts}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)

    def test_log_since_with_datetime(self):
        if False:
            return 10
        ts = 809222400
        time = datetime.datetime.utcfromtimestamp(ts)
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=False, since=time)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 0, 'stderr': 1, 'stdout': 1, 'tail': 'all', 'since': ts}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=False)

    def test_log_since_with_invalid_value_raises_error(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container):
            with pytest.raises(docker.errors.InvalidArgument):
                self.client.logs(fake_api.FAKE_CONTAINER_ID, stream=False, follow=False, since='42.42')

    def test_log_tty(self):
        if False:
            for i in range(10):
                print('nop')
        m = mock.Mock()
        with mock.patch('docker.api.client.APIClient.inspect_container', fake_inspect_container_tty):
            with mock.patch('docker.api.client.APIClient._stream_raw_result', m):
                self.client.logs(fake_api.FAKE_CONTAINER_ID, follow=True, stream=True)
        assert m.called
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/logs', params={'timestamps': 0, 'follow': 1, 'stderr': 1, 'stdout': 1, 'tail': 'all'}, timeout=DEFAULT_TIMEOUT_SECONDS, stream=True)

    def test_diff(self):
        if False:
            return 10
        self.client.diff(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/changes', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_diff_with_dict_instead_of_id(self):
        if False:
            return 10
        self.client.diff({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/changes', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_port(self):
        if False:
            i = 10
            return i + 15
        self.client.port({'Id': fake_api.FAKE_CONTAINER_ID}, 1111)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/json', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_stop_container(self):
        if False:
            i = 10
            return i + 15
        timeout = 2
        self.client.stop(fake_api.FAKE_CONTAINER_ID, timeout=timeout)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/stop', params={'t': timeout}, timeout=DEFAULT_TIMEOUT_SECONDS + timeout)

    def test_stop_container_with_dict_instead_of_id(self):
        if False:
            while True:
                i = 10
        timeout = 2
        self.client.stop({'Id': fake_api.FAKE_CONTAINER_ID}, timeout=timeout)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/stop', params={'t': timeout}, timeout=DEFAULT_TIMEOUT_SECONDS + timeout)

    def test_pause_container(self):
        if False:
            print('Hello World!')
        self.client.pause(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/pause', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_unpause_container(self):
        if False:
            print('Hello World!')
        self.client.unpause(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/unpause', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_kill_container(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.kill(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/kill', params={}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_kill_container_with_dict_instead_of_id(self):
        if False:
            i = 10
            return i + 15
        self.client.kill({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/kill', params={}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_kill_container_with_signal(self):
        if False:
            return 10
        self.client.kill(fake_api.FAKE_CONTAINER_ID, signal=signal.SIGTERM)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/kill', params={'signal': signal.SIGTERM}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_restart_container(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.restart(fake_api.FAKE_CONTAINER_ID, timeout=2)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/restart', params={'t': 2}, timeout=DEFAULT_TIMEOUT_SECONDS + 2)

    def test_restart_container_with_dict_instead_of_id(self):
        if False:
            i = 10
            return i + 15
        self.client.restart({'Id': fake_api.FAKE_CONTAINER_ID}, timeout=2)
        fake_request.assert_called_with('POST', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/restart', params={'t': 2}, timeout=DEFAULT_TIMEOUT_SECONDS + 2)

    def test_remove_container(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.remove_container(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('DELETE', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID, params={'v': False, 'link': False, 'force': False}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_remove_container_with_dict_instead_of_id(self):
        if False:
            i = 10
            return i + 15
        self.client.remove_container({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('DELETE', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID, params={'v': False, 'link': False, 'force': False}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_export(self):
        if False:
            print('Hello World!')
        self.client.export(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/export', stream=True, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_export_with_dict_instead_of_id(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.export({'Id': fake_api.FAKE_CONTAINER_ID})
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/export', stream=True, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_inspect_container(self):
        if False:
            print('Hello World!')
        self.client.inspect_container(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/json', timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_inspect_container_undefined_id(self):
        if False:
            while True:
                i = 10
        for arg in (None, '', {True: True}):
            with pytest.raises(docker.errors.NullResource) as excinfo:
                self.client.inspect_container(arg)
            assert excinfo.value.args[0] == 'Resource ID was not provided'

    def test_container_stats(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.stats(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/stats', stream=True, timeout=60, params={'stream': True})

    def test_container_stats_without_streaming(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.stats(fake_api.FAKE_CONTAINER_ID, stream=False)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/stats', timeout=60, params={'stream': False})

    def test_container_stats_with_one_shot(self):
        if False:
            while True:
                i = 10
        self.client.stats(fake_api.FAKE_CONTAINER_ID, stream=False, one_shot=True)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/stats', timeout=60, params={'stream': False, 'one-shot': True})

    def test_container_top(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.top(fake_api.FAKE_CONTAINER_ID)
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/top', params={}, timeout=DEFAULT_TIMEOUT_SECONDS)

    def test_container_top_with_psargs(self):
        if False:
            while True:
                i = 10
        self.client.top(fake_api.FAKE_CONTAINER_ID, 'waux')
        fake_request.assert_called_with('GET', url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/top', params={'ps_args': 'waux'}, timeout=DEFAULT_TIMEOUT_SECONDS)

    @requires_api_version('1.22')
    def test_container_update(self):
        if False:
            return 10
        self.client.update_container(fake_api.FAKE_CONTAINER_ID, mem_limit='2k', cpu_shares=124, blkio_weight=345)
        args = fake_request.call_args
        assert args[0][1] == url_prefix + 'containers/' + fake_api.FAKE_CONTAINER_ID + '/update'
        assert json.loads(args[1]['data']) == {'Memory': 2 * 1024, 'CpuShares': 124, 'BlkioWeight': 345}
        assert args[1]['headers']['Content-Type'] == 'application/json'