import shutil
import pytest
import salt.loader
import salt.roster.ansible as ansible
from tests.support.mock import patch
pytestmark = [pytest.mark.skip_if_binaries_missing('ansible-inventory')]

@pytest.fixture
def roster_opts():
    if False:
        print('Hello World!')
    return {'roster_defaults': {'passwd': 'test123'}}

@pytest.fixture
def configure_loader_modules(temp_salt_master, roster_opts):
    if False:
        return 10
    opts = temp_salt_master.config.copy()
    utils = salt.loader.utils(opts, whitelist=['json', 'stringutils', 'ansible'])
    runner = salt.loader.runner(opts, utils=utils, whitelist=['salt'])
    return {ansible: {'__utils__': utils, '__opts__': roster_opts, '__runner__': runner}}

@pytest.fixture
def expected_targets_return():
    if False:
        print('Hello World!')
    return {'host1': {'host': 'host1', 'passwd': 'test123', 'minion_opts': {'escape_pods': 2, 'halon_system_timeout': 30, 'self_destruct_countdown': 60, 'some_server': 'foo.southeast.example.com'}}, 'host2': {'host': 'host2', 'passwd': 'test123', 'minion_opts': {'escape_pods': 2, 'halon_system_timeout': 30, 'self_destruct_countdown': 60, 'some_server': 'foo.southeast.example.com'}}, 'host3': {'host': 'host3', 'passwd': 'test123', 'minion_opts': {'escape_pods': 2, 'halon_system_timeout': 30, 'self_destruct_countdown': 60, 'some_server': 'foo.southeast.example.com'}}}

@pytest.fixture
def expected_docs_targets_return():
    if False:
        for i in range(10):
            print('nop')
    return {'home': {'passwd': 'password', 'sudo': 'password', 'host': '12.34.56.78', 'port': 23, 'user': 'gtmanfred', 'minion_opts': {'http_port': 80}}, 'salt.gtmanfred.com': {'passwd': 'password', 'sudo': 'password', 'host': '127.0.0.1', 'port': 22, 'user': 'gtmanfred', 'minion_opts': {'http_port': 80}}}

@pytest.fixture(scope='module')
def roster_dir(tmp_path_factory):
    if False:
        return 10
    dpath = tmp_path_factory.mktemp('roster')
    roster_py_contents = '\n    #!/usr/bin/env python\n\n    import json\n    import sys\n\n    inventory = {\n        "usa": {"children": ["southeast"]},\n        "southeast": {\n            "children": ["atlanta", "raleigh"],\n            "vars": {\n                "some_server": "foo.southeast.example.com",\n                "halon_system_timeout": 30,\n                "self_destruct_countdown": 60,\n                "escape_pods": 2,\n            },\n        },\n        "raleigh": ["host2", "host3"],\n        "atlanta": ["host1", "host2"],\n    }\n    hostvars = {"host1": {}, "host2": {}, "host3": {}}\n\n    if "--host" in sys.argv:\n        print(json.dumps(hostvars.get(sys.argv[-1], {})))\n    if "--list" in sys.argv:\n        print(json.dumps(inventory))\n    '
    roster_ini_contents = '\n    [atlanta]\n    host1\n    host2\n\n    [raleigh]\n    host2\n    host3\n\n    [southeast:children]\n    atlanta\n    raleigh\n\n    [southeast:vars]\n    some_server=foo.southeast.example.com\n    halon_system_timeout=30\n    self_destruct_countdown=60\n    escape_pods=2\n\n    [usa:children]\n    southeast\n    '
    roster_yaml_contents = '\n    atlanta:\n      hosts:\n        host1:\n        host2:\n    raleigh:\n      hosts:\n        host2:\n        host3:\n    southeast:\n      children:\n        atlanta:\n        raleigh:\n      vars:\n        some_server: foo.southeast.example.com\n        halon_system_timeout: 30\n        self_destruct_countdown: 60\n        escape_pods: 2\n    usa:\n      children:\n        southeast:\n    '
    docs_ini_contents = "\n    [servers]\n    salt.gtmanfred.com ansible_ssh_user=gtmanfred ansible_ssh_host=127.0.0.1 ansible_ssh_port=22 ansible_ssh_pass='password' ansible_sudo_pass='password'\n\n    [desktop]\n    home ansible_ssh_user=gtmanfred ansible_ssh_host=12.34.56.78 ansible_ssh_port=23 ansible_ssh_pass='password' ansible_sudo_pass='password'\n\n    [computers:children]\n    desktop\n    servers\n\n    [computers:vars]\n    http_port=80\n    "
    docs_script_contents = '\n    #!/bin/bash\n    echo \'{\n        "servers": [\n            "salt.gtmanfred.com"\n        ],\n        "desktop": [\n            "home"\n        ],\n        "computers": {\n            "hosts": [],\n            "children": [\n                "desktop",\n                "servers"\n            ],\n            "vars": {\n                "http_port": 80\n            }\n        },\n        "_meta": {\n            "hostvars": {\n                "salt.gtmanfred.com": {\n                    "ansible_ssh_user": "gtmanfred",\n                    "ansible_ssh_host": "127.0.0.1",\n                    "ansible_sudo_pass": "password",\n                    "ansible_ssh_pass": "password",\n                    "ansible_ssh_port": 22\n                },\n                "home": {\n                    "ansible_ssh_user": "gtmanfred",\n                    "ansible_ssh_host": "12.34.56.78",\n                    "ansible_sudo_pass": "password",\n                    "ansible_ssh_pass": "password",\n                    "ansible_ssh_port": 23\n                }\n            }\n        }\n    }\'\n    '
    with pytest.helpers.temp_file('roster.py', roster_py_contents, directory=dpath) as py_roster:
        py_roster.chmod(493)
        with pytest.helpers.temp_file('roster.ini', roster_ini_contents, directory=dpath), pytest.helpers.temp_file('roster.yml', roster_yaml_contents, directory=dpath), pytest.helpers.temp_file('roster-docs.ini', docs_ini_contents, directory=dpath):
            with pytest.helpers.temp_file('roster-docs.sh', docs_script_contents, directory=dpath) as script_roster:
                script_roster.chmod(493)
                try:
                    yield dpath
                finally:
                    shutil.rmtree(str(dpath), ignore_errors=True)

@pytest.mark.parametrize('which_value', [False, None])
def test_virtual_returns_False_if_ansible_inventory_doesnt_exist(which_value):
    if False:
        return 10
    with patch('salt.utils.path.which', autospec=True, return_value=which_value):
        assert ansible.__virtual__() == (False, 'Install `ansible` to use inventory')

def test_ini(roster_opts, roster_dir, expected_targets_return):
    if False:
        return 10
    roster_opts['roster_file'] = str(roster_dir / 'roster.ini')
    with patch.dict(ansible.__opts__, roster_opts):
        ret = ansible.targets('*')
        assert ret == expected_targets_return

def test_yml(roster_opts, roster_dir, expected_targets_return):
    if False:
        i = 10
        return i + 15
    roster_opts['roster_file'] = str(roster_dir / 'roster.yml')
    with patch.dict(ansible.__opts__, roster_opts):
        ret = ansible.targets('*')
        assert ret == expected_targets_return

def test_script(roster_opts, roster_dir, expected_targets_return):
    if False:
        print('Hello World!')
    roster_opts['roster_file'] = str(roster_dir / 'roster.py')
    with patch.dict(ansible.__opts__, roster_opts):
        ret = ansible.targets('*')
        assert ret == expected_targets_return

def test_docs_ini(roster_opts, roster_dir, expected_docs_targets_return):
    if False:
        while True:
            i = 10
    roster_opts['roster_file'] = str(roster_dir / 'roster-docs.ini')
    with patch.dict(ansible.__opts__, roster_opts):
        ret = ansible.targets('*')
        assert ret == expected_docs_targets_return

def test_docs_script(roster_opts, roster_dir, expected_docs_targets_return):
    if False:
        i = 10
        return i + 15
    roster_opts['roster_file'] = str(roster_dir / 'roster-docs.sh')
    with patch.dict(ansible.__opts__, roster_opts):
        ret = ansible.targets('*')
        assert ret == expected_docs_targets_return