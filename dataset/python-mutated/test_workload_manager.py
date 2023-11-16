import os
from unittest.mock import MagicMock
from kubernetes import client
from mage_ai.cluster_manager.errors import ConfigurationError
from mage_ai.cluster_manager.kubernetes.workload_manager import WorkloadManager
from mage_ai.tests.base_test import TestCase

class WorkloadManagerTests(TestCase):

    def setUp(self):
        if False:
            return 10
        self.mock_core_client = MagicMock()
        self.mock_apps_client = MagicMock()
        self.mock_networking_client = MagicMock()
        client.CoreV1Api = MagicMock(return_value=self.mock_core_client)
        client.AppsV1Api = MagicMock(return_value=self.mock_apps_client)
        client.NetworkingV1Api = MagicMock(return_value=self.mock_networking_client)
        self.workload_manager = WorkloadManager()

    def test_configure_pre_start(self):
        if False:
            i = 10
            return i + 15
        pre_start_script_path = os.path.join(self.repo_path, 'pre_start_test_script.py')
        with open(pre_start_script_path, 'w') as f:
            f.write("\ndef get_custom_configs(config):\n    current_env = config.get('env', [])\n    current_env.append({\n        'name': 'RANDOM_ENV_VAR',\n        'value': 'HI HELLO',\n    })\n    config['env'] = current_env\n    return config\n")
        self.workload_manager.create_hooks_config_map('test', pre_start_script_path=pre_start_script_path, mage_container_config={'name': 'test-container', 'image': 'mageai/mageai:latest', 'ports': [{'containerPort': 6789, 'name': 'web'}]})
        self.mock_core_client.create_namespaced_config_map.assert_called_once()
        os.remove(pre_start_script_path)

    def test_configure_pre_start_invalid_wrong_function_name(self):
        if False:
            return 10
        pre_start_script_path = os.path.join(self.repo_path, 'pre_start_test_script.py')
        with open(pre_start_script_path, 'w') as f:
            f.write("\ndef wrong_function_name(config):\n    current_env = config.get('env', [])\n    current_env.append({\n        'name': 'RANDOM_ENV_VAR',\n        'value': 'HI HELLO',\n    })\n    config['env'] = current_env\n    return config\n")
        with self.assertRaises(Exception) as context:
            self.workload_manager.create_hooks_config_map('test', pre_start_script_path=pre_start_script_path, mage_container_config={'name': 'test-container', 'image': 'mageai/mageai:latest', 'ports': [{'containerPort': 6789, 'name': 'web'}]})
            self.assertTrue('Could not find get_custom_configs function' in str(context.exception))
            self.mock_core_client.create_namespaced_config_map.assert_not_called()
        os.remove(pre_start_script_path)

    def test_configure_pre_start_invalid_empty_container_config(self):
        if False:
            for i in range(10):
                print('nop')
        pre_start_script_path = os.path.join(self.repo_path, 'pre_start_test_script.py')
        with open(pre_start_script_path, 'w') as f:
            f.write("\ndef wrong_function_name(config):\n    current_env = config.get('env', [])\n    current_env.append({\n        'name': 'RANDOM_ENV_VAR',\n        'value': 'HI HELLO',\n    })\n    config['env'] = current_env\n    return config\n")
        with self.assertRaises(ConfigurationError) as context:
            self.workload_manager.create_hooks_config_map('test', pre_start_script_path=pre_start_script_path, mage_container_config=dict())
            self.assertTrue('The container config can not be empty' in str(context.exception))
            self.mock_core_client.create_namespaced_config_map.assert_not_called()
        os.remove(pre_start_script_path)