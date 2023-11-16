from __future__ import annotations
from ansible.module_utils.facts.hardware import linux
from .linux_data import CPU_INFO_TEST_SCENARIOS

def test_get_cpu_info(mocker):
    if False:
        while True:
            i = 10
    module = mocker.Mock()
    inst = linux.LinuxHardware(module)
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('os.access', return_value=True)
    for test in CPU_INFO_TEST_SCENARIOS:
        mocker.patch('ansible.module_utils.facts.hardware.linux.get_file_lines', side_effect=[[], test['cpuinfo']])
        mocker.patch('os.sched_getaffinity', create=True, return_value=test['sched_getaffinity'])
        module.run_command.return_value = (0, test['nproc_out'], '')
        collected_facts = {'ansible_architecture': test['architecture']}
        assert test['expected_result'] == inst.get_cpu_facts(collected_facts=collected_facts)

def test_get_cpu_info_nproc(mocker):
    if False:
        for i in range(10):
            print('nop')
    module = mocker.Mock()
    inst = linux.LinuxHardware(module)
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('os.access', return_value=True)
    for test in CPU_INFO_TEST_SCENARIOS:
        mocker.patch('ansible.module_utils.facts.hardware.linux.get_file_lines', side_effect=[[], test['cpuinfo']])
        mocker.patch('os.sched_getaffinity', create=True, side_effect=AttributeError)
        mocker.patch('ansible.module_utils.facts.hardware.linux.get_bin_path', return_value='/usr/bin/nproc')
        module.run_command.return_value = (0, test['nproc_out'], '')
        collected_facts = {'ansible_architecture': test['architecture']}
        assert test['expected_result'] == inst.get_cpu_facts(collected_facts=collected_facts)

def test_get_cpu_info_missing_arch(mocker):
    if False:
        while True:
            i = 10
    module = mocker.Mock()
    inst = linux.LinuxHardware(module)
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('os.access', return_value=True)
    for test in CPU_INFO_TEST_SCENARIOS:
        mocker.patch('ansible.module_utils.facts.hardware.linux.get_file_lines', side_effect=[[], test['cpuinfo']])
        mocker.patch('os.sched_getaffinity', create=True, return_value=test['sched_getaffinity'])
        module.run_command.return_value = (0, test['nproc_out'], '')
        test_result = inst.get_cpu_facts()
        if test['architecture'].startswith(('armv', 'aarch', 'ppc', 's390')):
            assert test['expected_result'] != test_result
        else:
            assert test['expected_result'] == test_result