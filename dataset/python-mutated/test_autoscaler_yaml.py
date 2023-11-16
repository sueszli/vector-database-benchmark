import copy
import logging
import os
import sys
import tempfile
from typing import Dict, Any
import unittest
import urllib
from unittest.mock import MagicMock, Mock, patch
import jsonschema
import pytest
import yaml
from click.exceptions import ClickException
import mock
from ray._private.test_utils import load_test_config, recursive_fnmatch
from ray.autoscaler._private._azure.config import _configure_key_pair as _azure_configure_key_pair
from ray.autoscaler._private.gcp import config as gcp_config
from ray.autoscaler._private.providers import _NODE_PROVIDERS
from ray.autoscaler._private.util import fill_node_type_min_max_workers, merge_setup_commands, prepare_config, validate_config
RAY_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATHS = recursive_fnmatch(os.path.join(RAY_PATH, 'autoscaler'), '*.yaml')
CONFIG_PATHS += recursive_fnmatch(os.path.join(RAY_PATH, 'tune', 'examples'), '*.yaml')
EXPECTED_LOCAL_CONFIG_STR = '\ncluster_name: minimal-manual\nprovider:\n  head_ip: xxx.yyy\n  type: local\n  worker_ips:\n  - aaa.bbb\n  - ccc.ddd\n  - eee.fff\nauth:\n  ssh_private_key: ~/.ssh/id_rsa\n  ssh_user: user\ndocker: {}\nmax_workers: 3\navailable_node_types:\n  local.cluster.node:\n    max_workers: 3\n    min_workers: 3\n    node_config: {}\n    resources: {}\nhead_node_type: local.cluster.node\nhead_start_ray_commands:\n- ray stop\n- ulimit -c unlimited; ray start --head --port=6379 --autoscaling-config=~/ray_bootstrap_config.yaml\nworker_start_ray_commands:\n- ray stop\n- ray start --address=$RAY_HEAD_IP:6379\ncluster_synced_files: []\nidle_timeout_minutes: 5\nupscaling_speed: 1.0\nfile_mounts: {}\nfile_mounts_sync_continuously: false\nhead_setup_commands: []\ninitialization_commands: []\nrsync_exclude: []\nrsync_filter: []\nsetup_commands: []\nworker_setup_commands: []\n'

def fake_fillout_available_node_types_resources(config: Dict[str, Any]) -> None:
    if False:
        return 10
    'A cheap way to fill out the resources field (the same way a node\n    provider would autodetect them) as far as schema validation is concerned.'
    available_node_types = config.get('available_node_types', {})
    for value in available_node_types.values():
        value['resources'] = value.get('resources', {'filler': 1})

class AutoscalingConfigTest(unittest.TestCase):

    def testValidateDefaultConfig(self):
        if False:
            i = 10
            return i + 15
        for config_path in CONFIG_PATHS:
            try:
                if os.path.join('aws', 'example-multi-node-type.yaml') in config_path:
                    continue
                if 'local' in config_path:
                    continue
                if 'fake_multi_node' in config_path:
                    continue
                if 'kuberay' in config_path:
                    continue
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                config = prepare_config(config)
                if config['provider']['type'] == 'aws':
                    fake_fillout_available_node_types_resources(config)
                validate_config(config)
            except Exception:
                logging.exception('')
                self.fail(f'Config {config_path} did not pass validation test!')

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testValidateDefaultConfigMinMaxWorkers(self):
        if False:
            for i in range(10):
                print('nop')
        aws_config_path = os.path.join(RAY_PATH, 'autoscaler/aws/example-multi-node-type.yaml')
        with open(aws_config_path) as f:
            config = yaml.safe_load(f)
        config = prepare_config(config)
        for node_type in config['available_node_types']:
            config['available_node_types'][node_type]['resources'] = config['available_node_types'][node_type].get('resources', {})
        try:
            validate_config(config)
        except Exception:
            self.fail('Config did not pass validation test!')
        config['max_workers'] = 0
        with pytest.raises(ValueError):
            validate_config(config)
        config['max_workers'] = 1
        try:
            validate_config(config)
        except Exception:
            self.fail('Config did not pass validation test!')

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testValidateDefaultConfigAWSMultiNodeTypes(self):
        if False:
            while True:
                i = 10
        aws_config_path = os.path.join(RAY_PATH, 'autoscaler/aws/example-multi-node-type.yaml')
        with open(aws_config_path) as f:
            config = yaml.safe_load(f)
        new_config = copy.deepcopy(config)
        new_config['available_node_types'] = {'cpu_4_ondemand': new_config['available_node_types']['cpu_4_ondemand'], 'cpu_16_spot': new_config['available_node_types']['cpu_16_spot'], 'gpu_8_ondemand': new_config['available_node_types']['gpu_8_ondemand'], 'neuron_core_inf_1_ondemand': {'node_config': {'InstanceType': 'inf2.xlarge', 'ImageId': 'latest_dlami'}, 'max_workers': 2}}
        orig_new_config = copy.deepcopy(new_config)
        expected_available_node_types = orig_new_config['available_node_types']
        expected_available_node_types['cpu_4_ondemand']['resources'] = {'CPU': 4}
        expected_available_node_types['cpu_16_spot']['resources'] = {'CPU': 16, 'memory': 48103633715, 'Custom1': 1, 'is_spot': 1}
        expected_available_node_types['gpu_8_ondemand']['resources'] = {'CPU': 32, 'memory': 183395103539, 'GPU': 4, 'accelerator_type:V100': 1}
        expected_available_node_types['neuron_core_inf_1_ondemand']['resources'] = {'CPU': 4, 'memory': 12025908428, 'neuron_cores': 2, 'accelerator_type:aws-neuron-core': 1}
        expected_available_node_types['cpu_16_spot']['min_workers'] = 0
        expected_available_node_types['gpu_8_ondemand']['min_workers'] = 0
        expected_available_node_types['neuron_core_inf_1_ondemand']['min_workers'] = 0
        boto3_dict = {'InstanceTypes': [{'InstanceType': 'm4.xlarge', 'VCpuInfo': {'DefaultVCpus': 4}, 'MemoryInfo': {'SizeInMiB': 16384}}, {'InstanceType': 'm4.4xlarge', 'VCpuInfo': {'DefaultVCpus': 16}, 'MemoryInfo': {'SizeInMiB': 65536}}, {'InstanceType': 'p3.8xlarge', 'VCpuInfo': {'DefaultVCpus': 32}, 'MemoryInfo': {'SizeInMiB': 249856}, 'GpuInfo': {'Gpus': [{'Name': 'V100', 'Count': 4}]}}, {'InstanceType': 'inf2.xlarge', 'VCpuInfo': {'DefaultVCpus': 4}, 'MemoryInfo': {'SizeInMiB': 16384}, 'AcceleratorInfo': {'Accelerators': [{'Name': 'Inferentia', 'Count': 1}]}}]}
        describe_instance_types_mock = Mock()
        describe_instance_types_mock.describe_instance_types = MagicMock(return_value=boto3_dict)
        client_cache_mock = MagicMock(return_value=describe_instance_types_mock)
        with patch.multiple('ray.autoscaler._private.aws.node_provider', client_cache=client_cache_mock):
            new_config = prepare_config(new_config)
            importer = _NODE_PROVIDERS.get(new_config['provider']['type'])
            provider_cls = importer(new_config['provider'])
            try:
                new_config = provider_cls.fillout_available_node_types_resources(new_config)
                validate_config(new_config)
                assert expected_available_node_types == new_config['available_node_types']
            except Exception:
                self.fail('Config did not pass multi node types auto fill test!')

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testValidateLocal(self):
        if False:
            return 10
        '\n        Tests local node provider config validation for the most common use\n        case of bootstrapping a cluster at a static set of ips.\n        '
        local_config_path = os.path.join(RAY_PATH, 'autoscaler/local/example-minimal-manual.yaml')
        base_config = yaml.safe_load(open(local_config_path).read())
        base_config['provider']['head_ip'] = 'xxx.yyy'
        base_config['provider']['worker_ips'] = ['aaa.bbb', 'ccc.ddd', 'eee.fff']
        base_config['auth']['ssh_user'] = 'user'
        base_config['auth']['ssh_private_key'] = '~/.ssh/id_rsa'
        test_prepare_config = copy.deepcopy(base_config)
        prepared_config = prepare_config(test_prepare_config)
        try:
            validate_config(prepared_config)
        except Exception:
            self.fail('Failed to validate local/example-minimal-manual.yaml')
        expected_prepared = yaml.safe_load(EXPECTED_LOCAL_CONFIG_STR)
        assert prepared_config == expected_prepared
        no_worker_config = copy.deepcopy(base_config)
        del no_worker_config['provider']['worker_ips']
        with pytest.raises(ClickException):
            prepare_config(no_worker_config)
        no_head_config = copy.deepcopy(base_config)
        del no_head_config['provider']['head_ip']
        with pytest.raises(ClickException):
            prepare_config(no_head_config)
        for field in ('head_node', 'worker_nodes', 'available_node_types'):
            faulty_config = copy.deepcopy(base_config)
            faulty_config[field] = "This field shouldn't be in here."
            with pytest.raises(ClickException):
                prepare_config(faulty_config)
        too_many_workers_config = copy.deepcopy(base_config)
        too_many_workers_config['max_workers'] = 10
        too_many_workers_config['min_workers'] = 10
        prepared_config = prepare_config(too_many_workers_config)
        assert prepared_config == expected_prepared
        not_enough_workers_config = copy.deepcopy(base_config)
        not_enough_workers_config['max_workers'] = 0
        not_enough_workers_config['min_workers'] = 0
        with mock.patch('ray.autoscaler._private.local.config.cli_logger.warning') as warning:
            prepared_config = prepare_config(not_enough_workers_config)
            warning.assert_called_with('The value of `max_workers` supplied (0) is less than the number of available worker ips (3). At most 0 Ray worker nodes will connect to the cluster.')
        expected_prepared = yaml.safe_load(EXPECTED_LOCAL_CONFIG_STR)
        expected_prepared['max_workers'] = 0
        expected_prepared['available_node_types']['local.cluster.node']['max_workers'] = 0
        expected_prepared['available_node_types']['local.cluster.node']['min_workers'] = 0
        assert prepared_config == expected_prepared

    def testValidateNetworkConfigForBackwardsCompatibility(self):
        if False:
            while True:
                i = 10
        web_yaml = 'https://raw.githubusercontent.com/ray-project/ray/master/python/ray/autoscaler/aws/example-full.yaml'
        response = urllib.request.urlopen(web_yaml, timeout=5)
        content = response.read()
        with tempfile.TemporaryFile() as f:
            f.write(content)
            f.seek(0)
            config = yaml.safe_load(f)
        config = prepare_config(config)
        try:
            validate_config(config)
        except Exception:
            self.fail('Config did not pass validation test!')

    def _test_invalid_config(self, config_path):
        if False:
            while True:
                i = 10
        with open(os.path.join(RAY_PATH, config_path)) as f:
            config = yaml.safe_load(f)
        try:
            validate_config(config)
            self.fail('Expected validation to fail for {}'.format(config_path))
        except jsonschema.ValidationError:
            pass

    @unittest.skipIf(sys.platform == 'win32', 'Failing on Windows.')
    def testInvalidConfig(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid_config(os.path.join('tests', 'additional_property.yaml'))

    @unittest.skipIf(sys.platform == 'win32', 'Failing on Windows.')
    def testValidateCustomSecurityGroupConfig(self):
        if False:
            while True:
                i = 10
        aws_config_path = os.path.join(RAY_PATH, 'autoscaler/aws/example-minimal.yaml')
        with open(aws_config_path) as f:
            config = yaml.safe_load(f)
        ip_permissions = [{'FromPort': port, 'ToPort': port, 'IpProtocol': 'TCP', 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]} for port in [80, 443, 8265]]
        config['provider'].update({'security_group': {'IpPermissions': ip_permissions}})
        config = prepare_config(copy.deepcopy(config))
        try:
            validate_config(config)
            assert config['provider']['security_group']['IpPermissions'] == ip_permissions
        except Exception:
            self.fail('Failed to validate config with security group in bound rules!')
        group_name = 'test_security_group_name'
        config['provider']['security_group'].update({'GroupName': group_name})
        try:
            validate_config(config)
            assert config['provider']['security_group']['GroupName'] == group_name
        except Exception:
            self.fail('Failed to validate config with security group name!')

    def testMaxWorkerDefault(self):
        if False:
            i = 10
            return i + 15
        config = load_test_config('test_multi_node.yaml')
        node_types = config['available_node_types']
        assert 'max_workers' not in node_types['worker_node_max_unspecified']
        assert 'max_workers' in node_types['worker_node_max_specified']
        prepared_config = prepare_config(config)
        prepared_node_types = prepared_config['available_node_types']
        assert node_types['worker_node_max_specified']['max_workers'] == prepared_node_types['worker_node_max_specified']['max_workers'] == 3
        assert config['max_workers'] == prepared_node_types['worker_node_max_unspecified']['max_workers'] == 5
        config = load_test_config('test_multi_node.yaml')
        del config['max_workers']
        node_types = config['available_node_types']
        assert 'max_workers' not in node_types['worker_node_max_unspecified']
        assert 'max_workers' in node_types['worker_node_max_specified']
        prepared_config = prepare_config(config)
        prepared_node_types = prepared_config['available_node_types']
        assert node_types['worker_node_max_specified']['max_workers'] == prepared_node_types['worker_node_max_specified']['max_workers'] == 3
        assert prepared_config['max_workers'] == prepared_node_types['worker_node_max_unspecified']['max_workers'] == 2

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testExampleFull(self):
        if False:
            while True:
                i = 10
        '\n        Test that example-full yamls are unmodified by prepared_config,\n        except possibly by having setup_commands merged and\n        default per-node max/min workers set.\n        '
        providers = ['aws', 'gcp', 'azure']
        for provider in providers:
            path = os.path.join(RAY_PATH, 'autoscaler', provider, 'example-full.yaml')
            config = yaml.safe_load(open(path).read())
            config_copy = copy.deepcopy(config)
            merge_setup_commands(config_copy)
            fill_node_type_min_max_workers(config_copy)
            assert config_copy == prepare_config(config)

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testAzureKeyPair(self):
        if False:
            for i in range(10):
                print('nop')
        azure_config_path = os.path.join(RAY_PATH, 'autoscaler/azure/example-full.yaml')
        azure_config = yaml.safe_load(open(azure_config_path))
        azure_config['auth']['ssh_user'] = 'default_user'
        with tempfile.NamedTemporaryFile() as pub_key, tempfile.NamedTemporaryFile() as priv_key:
            pub_key.write(b'PUBLICKEY')
            pub_key.flush()
            priv_key.write(b'PRIVATEKEY')
            priv_key.flush()
            azure_config['auth']['ssh_private_key'] = priv_key.name
            azure_config['auth']['ssh_public_key'] = pub_key.name
            modified_config = _azure_configure_key_pair(azure_config)
        for node_type in modified_config['available_node_types'].values():
            assert node_type['node_config']['azure_arm_parameters']['adminUsername'] == 'default_user'
            assert node_type['node_config']['azure_arm_parameters']['publicKey'] == 'PUBLICKEY'

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testGCPSubnets(self):
        if False:
            print('Hello World!')
        "Validates gcp _configure_subnet logic.\n\n        Checks that _configure_subnet fills default networkInterfaces data for\n        each node type that doesn't specify networkInterfaces.\n\n        Checks that _list_subnets is not called if all node types specify\n        networkInterfaces.\n        "
        path = os.path.join(RAY_PATH, 'autoscaler', 'gcp', 'example-full.yaml')
        config = yaml.safe_load(open(path).read())
        config_subnets_configured = copy.deepcopy(config)
        config_subnets_worker_configured = copy.deepcopy(config)
        config_subnets_head_configured = copy.deepcopy(config)
        config_subnets_no_type_configured = copy.deepcopy(config)
        config_subnets_configured['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] = 'mock_interfaces'
        config_subnets_configured['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] = 'mock_interfaces'
        config_subnets_worker_configured['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] = 'mock_interfaces'
        config_subnets_head_configured['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] = 'mock_interfaces'
        assert 'networkInterfaces' not in config_subnets_no_type_configured['available_node_types']['ray_head_default']['node_config']
        assert 'networkInterfaces' not in config_subnets_no_type_configured['available_node_types']['ray_worker_small']['node_config']
        config_subnets_configured_post = copy.deepcopy(config_subnets_configured)
        config_subnets_worker_configured_post = copy.deepcopy(config_subnets_worker_configured)
        config_subnets_head_configured_post = copy.deepcopy(config_subnets_head_configured)
        config_subnets_no_type_configured_post = copy.deepcopy(config_subnets_no_type_configured)
        list_subnets_counter = 0

        def mock_list_subnets(*args):
            if False:
                i = 10
                return i + 15
            nonlocal list_subnets_counter
            list_subnets_counter += 1
            return [{'selfLink': 'link'}]
        gcp_config._list_subnets = mock_list_subnets
        config_subnets_configured_post = gcp_config._configure_subnet(config_subnets_configured, compute='mock_compute')
        assert list_subnets_counter == 0
        config_subnets_worker_configured_post = gcp_config._configure_subnet(config_subnets_worker_configured, compute='mock_compute')
        assert list_subnets_counter == 1
        config_subnets_head_configured_post = gcp_config._configure_subnet(config_subnets_head_configured, compute='mock_compute')
        assert list_subnets_counter == 2
        config_subnets_no_type_configured_post = gcp_config._configure_subnet(config_subnets_no_type_configured, compute='mock_compute')
        assert list_subnets_counter == 3
        default_interfaces = [{'subnetwork': 'link', 'accessConfigs': [{'name': 'External NAT', 'type': 'ONE_TO_ONE_NAT'}]}]
        assert config_subnets_configured_post == config_subnets_configured
        assert config_subnets_configured_post['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] == 'mock_interfaces'
        assert config_subnets_configured_post['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] == 'mock_interfaces'
        assert config_subnets_worker_configured_post['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] == 'mock_interfaces'
        assert config_subnets_worker_configured_post['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] == default_interfaces
        assert config_subnets_head_configured_post['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] == default_interfaces
        assert config_subnets_head_configured_post['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] == 'mock_interfaces'
        assert config_subnets_no_type_configured_post['available_node_types']['ray_worker_small']['node_config']['networkInterfaces'] == default_interfaces
        assert config_subnets_no_type_configured_post['available_node_types']['ray_head_default']['node_config']['networkInterfaces'] == default_interfaces

    @pytest.mark.skipif(sys.platform.startswith('win'), reason='Fails on Windows.')
    def testFaultyResourceValidation(self):
        if False:
            while True:
                i = 10
        'Checks that schema validation catches invalid node type resource\n        field.\n\n        Demonstrates a fix in https://github.com/ray-project/ray/pull/16691.'
        path = os.path.join(RAY_PATH, 'autoscaler', 'aws', 'example-full.yaml')
        config = yaml.safe_load(open(path).read())
        node_type = config['available_node_types']['ray.head.default']
        node_type['resources'] = None
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validate_config(config)
        node_type['resources'] = {'CPU': 'a string is not valid here'}
        with pytest.raises(jsonschema.exceptions.ValidationError):
            validate_config(config)
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))