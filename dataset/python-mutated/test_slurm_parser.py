import pytest
import os
from ding.entry.cli_parsers import PLATFORM_PARSERS
from ding.entry.cli_parsers.slurm_parser import SlurmParser
slurm_parser = PLATFORM_PARSERS['slurm']

@pytest.fixture
def set_slurm_env():
    if False:
        while True:
            i = 10
    os.environ['SLURM_NTASKS'] = '6'
    os.environ['SLURM_NTASKS_PER_NODE'] = '3'
    os.environ['SLURM_NODELIST'] = 'SH-IDC1-10-5-38-[190,215]'
    os.environ['SLURM_PROCID'] = '3'
    os.environ['SLURMD_NODENAME'] = 'SH-IDC1-10-5-38-215'
    yield
    del os.environ['SLURM_NTASKS']
    del os.environ['SLURM_NTASKS_PER_NODE']
    del os.environ['SLURM_NODELIST']
    del os.environ['SLURM_PROCID']
    del os.environ['SLURMD_NODENAME']

@pytest.mark.unittest
@pytest.mark.usefixtures('set_slurm_env')
def test_slurm_parser():
    if False:
        print('Hello World!')
    platform_spec = {'tasks': [{'labels': 'league,collect', 'node_ids': 10}, {'labels': 'league,collect', 'node_ids': 11}, {'labels': 'evaluate', 'node_ids': 20, 'attach_to': '$node.10,$node.11'}, {'labels': 'learn', 'node_ids': 31, 'attach_to': '$node.10,$node.11,$node.20'}, {'labels': 'learn', 'node_ids': 32, 'attach_to': '$node.10,$node.11,$node.20'}, {'labels': 'learn', 'node_ids': 33, 'attach_to': '$node.10,$node.11,$node.20'}]}
    all_args = slurm_parser(platform_spec)
    assert all_args['labels'] == 'learn'
    assert all_args['address'] == 'SH-IDC1-10-5-38-215'
    assert all_args['ports'] == 15151
    assert all_args['node_ids'] == 31
    assert all_args['attach_to'] == 'tcp://SH-IDC1-10-5-38-190:15151,' + 'tcp://SH-IDC1-10-5-38-190:15152,' + 'tcp://SH-IDC1-10-5-38-190:15153'
    all_args = slurm_parser(None, topology='mesh', mq_type='nng')
    assert all_args['address'] == 'SH-IDC1-10-5-38-215'
    assert all_args['node_ids'] == 3
    assert all_args['parallel_workers'] == 1
    assert all_args['attach_to'] == 'tcp://SH-IDC1-10-5-38-190:15151,' + 'tcp://SH-IDC1-10-5-38-190:15152,' + 'tcp://SH-IDC1-10-5-38-190:15153'
    sp = SlurmParser(platform_spec)
    os.environ['SLURM_NODELIST'] = 'SH-IDC1-10-5-[38-40]'
    nodelist = sp._parse_node_list()
    assert nodelist == ['SH-IDC1-10-5-38', 'SH-IDC1-10-5-38', 'SH-IDC1-10-5-38', 'SH-IDC1-10-5-39', 'SH-IDC1-10-5-39', 'SH-IDC1-10-5-39', 'SH-IDC1-10-5-40', 'SH-IDC1-10-5-40', 'SH-IDC1-10-5-40']