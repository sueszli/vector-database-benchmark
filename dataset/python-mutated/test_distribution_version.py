from __future__ import annotations
import glob
import json
import os
import pytest
from itertools import product
import builtins
from ansible.module_utils.facts.system.distribution import DistributionFactCollector
TESTSETS = []
for datafile in glob.glob(os.path.join(os.path.dirname(__file__), 'fixtures/*.json')):
    with open(os.path.join(os.path.dirname(__file__), '%s' % datafile)) as f:
        TESTSETS.append(json.loads(f.read()))

@pytest.mark.parametrize('stdin, testcase', product([{}], TESTSETS), ids=lambda x: x.get('name'), indirect=['stdin'])
def test_distribution_version(am, mocker, testcase):
    if False:
        print('Hello World!')
    'tests the distribution parsing code of the Facts class\n\n    testsets have\n    * a name (for output/debugging only)\n    * input files that are faked\n      * those should be complete and also include "irrelevant" files that might be mistaken as coming from other distributions\n      * all files that are not listed here are assumed to not exist at all\n    * the output of ansible.module_utils.distro.linux_distribution() [called platform.dist() for historical reasons]\n    * results for the ansible variables distribution* and os_family\n\n    '

    def mock_get_file_content(fname, default=None, strip=True):
        if False:
            print('Hello World!')
        'give fake content if it exists, otherwise pretend the file is empty'
        data = default
        if fname in testcase['input']:
            print('faked %s for %s' % (fname, testcase['name']))
            data = testcase['input'][fname].strip()
        if strip and data is not None:
            data = data.strip()
        return data

    def mock_get_file_lines(fname, strip=True):
        if False:
            return 10
        'give fake lines if file exists, otherwise return empty list'
        data = mock_get_file_content(fname=fname, strip=strip)
        if data:
            return [data]
        return []

    def mock_get_uname(am, flags):
        if False:
            return 10
        if '-v' in flags:
            return testcase.get('uname_v', None)
        elif '-r' in flags:
            return testcase.get('uname_r', None)
        else:
            return None

    def mock_file_exists(fname, allow_empty=False):
        if False:
            print('Hello World!')
        if fname not in testcase['input']:
            return False
        if allow_empty:
            return True
        return bool(len(testcase['input'][fname]))

    def mock_platform_system():
        if False:
            print('Hello World!')
        return testcase.get('platform.system', 'Linux')

    def mock_platform_release():
        if False:
            return 10
        return testcase.get('platform.release', '')

    def mock_platform_version():
        if False:
            i = 10
            return i + 15
        return testcase.get('platform.version', '')

    def mock_distro_name():
        if False:
            i = 10
            return i + 15
        return testcase['distro']['name']

    def mock_distro_id():
        if False:
            print('Hello World!')
        return testcase['distro']['id']

    def mock_distro_version(best=False):
        if False:
            while True:
                i = 10
        if best:
            return testcase['distro']['version_best']
        return testcase['distro']['version']

    def mock_distro_codename():
        if False:
            for i in range(10):
                print('nop')
        return testcase['distro']['codename']

    def mock_distro_os_release_info():
        if False:
            return 10
        return testcase['distro']['os_release_info']

    def mock_distro_lsb_release_info():
        if False:
            for i in range(10):
                print('nop')
        return testcase['distro']['lsb_release_info']

    def mock_open(filename, mode='r'):
        if False:
            return 10
        if filename in testcase['input']:
            file_object = mocker.mock_open(read_data=testcase['input'][filename]).return_value
            file_object.__iter__.return_value = testcase['input'][filename].splitlines(True)
        else:
            file_object = real_open(filename, mode)
        return file_object

    def mock_os_path_is_file(filename):
        if False:
            while True:
                i = 10
        if filename in testcase['input']:
            return True
        return False

    def mock_run_command_output(v, command):
        if False:
            print('Hello World!')
        ret = (0, '', '')
        if 'command_output' in testcase:
            ret = (0, testcase['command_output'].get(command, ''), '')
        return ret
    mocker.patch('ansible.module_utils.facts.system.distribution.get_file_content', mock_get_file_content)
    mocker.patch('ansible.module_utils.facts.system.distribution.get_file_lines', mock_get_file_lines)
    mocker.patch('ansible.module_utils.facts.system.distribution.get_uname', mock_get_uname)
    mocker.patch('ansible.module_utils.facts.system.distribution._file_exists', mock_file_exists)
    mocker.patch('ansible.module_utils.distro.name', mock_distro_name)
    mocker.patch('ansible.module_utils.distro.id', mock_distro_id)
    mocker.patch('ansible.module_utils.distro.version', mock_distro_version)
    mocker.patch('ansible.module_utils.distro.codename', mock_distro_codename)
    mocker.patch('ansible.module_utils.common.sys_info.distro.os_release_info', mock_distro_os_release_info)
    mocker.patch('ansible.module_utils.common.sys_info.distro.lsb_release_info', mock_distro_lsb_release_info)
    mocker.patch('os.path.isfile', mock_os_path_is_file)
    mocker.patch('platform.system', mock_platform_system)
    mocker.patch('platform.release', mock_platform_release)
    mocker.patch('platform.version', mock_platform_version)
    mocker.patch('ansible.module_utils.basic.AnsibleModule.run_command', mock_run_command_output)
    real_open = builtins.open
    mocker.patch.object(builtins, 'open', new=mock_open)
    distro_collector = DistributionFactCollector()
    generated_facts = distro_collector.collect(am)
    for (key, val) in testcase['result'].items():
        assert key in generated_facts
        msg = 'Comparing value of %s on %s, should: %s, is: %s' % (key, testcase['name'], val, generated_facts[key])
        assert generated_facts[key] == val, msg