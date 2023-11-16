"""
    :codeauthor: Jochen Breuer <jbreuer@suse.de>
"""
import os
from contextlib import contextmanager
import pytest
import salt.utils.files
import salt.utils.platform
from salt.modules import kubernetesmod as kubernetes
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import Mock, patch
from tests.support.unit import TestCase

@contextmanager
def mock_kubernetes_library():
    if False:
        print('Hello World!')
    '\n    After fixing the bug in 1c821c0e77de58892c77d8e55386fac25e518c31,\n    it caused kubernetes._cleanup() to get called for virtually every\n    test, which blows up. This prevents that specific blow-up once\n    '
    with patch('salt.modules.kubernetesmod.kubernetes') as mock_kubernetes_lib:
        yield mock_kubernetes_lib

@pytest.mark.skipif(not kubernetes.HAS_LIBS, reason='Kubernetes client lib is not installed. Skipping test_kubernetes.py')
class KubernetesTestCase(TestCase, LoaderModuleMockMixin):
    """
    Test cases for salt.modules.kubernetesmod
    """

    def setup_loader_modules(self):
        if False:
            i = 10
            return i + 15
        return {kubernetes: {'__salt__': {}}}

    def test_nodes(self):
        if False:
            while True:
                i = 10
        '\n        Test node listing.\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.client.CoreV1Api.return_value = Mock(**{'list_node.return_value.to_dict.return_value': {'items': [{'metadata': {'name': 'mock_node_name'}}]}})
                self.assertEqual(kubernetes.nodes(), ['mock_node_name'])
                self.assertTrue(kubernetes.kubernetes.client.CoreV1Api().list_node().to_dict.called)

    def test_deployments(self):
        if False:
            print('Hello World!')
        '\n        Tests deployment listing.\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.client.ExtensionsV1beta1Api.return_value = Mock(**{'list_namespaced_deployment.return_value.to_dict.return_value': {'items': [{'metadata': {'name': 'mock_deployment_name'}}]}})
                self.assertEqual(kubernetes.deployments(), ['mock_deployment_name'])
                self.assertTrue(kubernetes.kubernetes.client.ExtensionsV1beta1Api().list_namespaced_deployment().to_dict.called)

    def test_services(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests services listing.\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.client.CoreV1Api.return_value = Mock(**{'list_namespaced_service.return_value.to_dict.return_value': {'items': [{'metadata': {'name': 'mock_service_name'}}]}})
                self.assertEqual(kubernetes.services(), ['mock_service_name'])
                self.assertTrue(kubernetes.kubernetes.client.CoreV1Api().list_namespaced_service().to_dict.called)

    def test_pods(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests pods listing.\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.client.CoreV1Api.return_value = Mock(**{'list_namespaced_pod.return_value.to_dict.return_value': {'items': [{'metadata': {'name': 'mock_pod_name'}}]}})
                self.assertEqual(kubernetes.pods(), ['mock_pod_name'])
                self.assertTrue(kubernetes.kubernetes.client.CoreV1Api().list_namespaced_pod().to_dict.called)

    def test_delete_deployments(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests deployment deletion\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch('salt.modules.kubernetesmod.show_deployment', Mock(return_value=None)):
                with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                    mock_kubernetes_lib.client.V1DeleteOptions = Mock(return_value='')
                    mock_kubernetes_lib.client.ExtensionsV1beta1Api.return_value = Mock(**{'delete_namespaced_deployment.return_value.to_dict.return_value': {'code': ''}})
                    self.assertEqual(kubernetes.delete_deployment('test'), {'code': 200})
                    self.assertTrue(kubernetes.kubernetes.client.ExtensionsV1beta1Api().delete_namespaced_deployment().to_dict.called)

    def test_create_deployments(self):
        if False:
            print('Hello World!')
        '\n        Tests deployment creation.\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.client.ExtensionsV1beta1Api.return_value = Mock(**{'create_namespaced_deployment.return_value.to_dict.return_value': {}})
                self.assertEqual(kubernetes.create_deployment('test', 'default', {}, {}, None, None, None), {})
                self.assertTrue(kubernetes.kubernetes.client.ExtensionsV1beta1Api().create_namespaced_deployment().to_dict.called)

    @staticmethod
    def settings(name, value=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test helper\n        :return: settings or default\n        '
        data = {'kubernetes.kubeconfig': '/home/testuser/.minikube/kubeconfig.cfg', 'kubernetes.context': 'minikube'}
        return data.get(name, value)

    def test_setup_kubeconfig_file(self):
        if False:
            return 10
        "\n        Test that the `kubernetes.kubeconfig` configuration isn't overwritten\n        :return:\n        "
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.config.load_kube_config = Mock()
                config = kubernetes._setup_conn()
                self.assertEqual(self.settings('kubernetes.kubeconfig'), config['kubeconfig'])

    def test_setup_kubeconfig_data_overwrite(self):
        if False:
            return 10
        '\n        Test that provided `kubernetes.kubeconfig` configuration is overwritten\n        by provided kubeconfig_data in the command\n        :return:\n        '
        with mock_kubernetes_library() as mock_kubernetes_lib:
            with patch.dict(kubernetes.__salt__, {'config.option': Mock(side_effect=self.settings)}):
                mock_kubernetes_lib.config.load_kube_config = Mock()
                config = kubernetes._setup_conn(kubeconfig_data='MTIzNDU2Nzg5MAo=', context='newcontext')
                check_path = os.path.join('/tmp', 'salt-kubeconfig-')
                if salt.utils.platform.is_windows():
                    check_path = os.path.join(os.environ.get('TMP'), 'salt-kubeconfig-')
                elif salt.utils.platform.is_darwin():
                    check_path = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'salt-kubeconfig-')
                self.assertTrue(config['kubeconfig'].lower().startswith(check_path.lower()))
                self.assertTrue(os.path.exists(config['kubeconfig']))
                with salt.utils.files.fopen(config['kubeconfig'], 'r') as kcfg:
                    self.assertEqual('1234567890\n', kcfg.read())
                kubernetes._cleanup(**config)

    def test_node_labels(self):
        if False:
            print('Hello World!')
        '\n        Test kubernetes.node_labels\n        :return:\n        '
        with patch('salt.modules.kubernetesmod.node') as mock_node:
            mock_node.return_value = {'metadata': {'labels': {'kubernetes.io/hostname': 'minikube', 'kubernetes.io/os': 'linux'}}}
            self.assertEqual(kubernetes.node_labels('minikube'), {'kubernetes.io/hostname': 'minikube', 'kubernetes.io/os': 'linux'})

    def test_adding_change_cause_annotation(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests adding a `kubernetes.io/change-cause` annotation just like\n        kubectl [apply|create|replace] --record does\n        :return:\n        '
        with patch('salt.modules.kubernetesmod.sys.argv', ['/usr/bin/salt-call', 'state.apply']) as mock_sys:
            func = getattr(kubernetes, '__dict_to_object_meta')
            data = func(name='test-pod', namespace='test', metadata={})
            self.assertEqual(data.name, 'test-pod')
            self.assertEqual(data.namespace, 'test')
            self.assertEqual(data.annotations, {'kubernetes.io/change-cause': '/usr/bin/salt-call state.apply'})
            test_metadata = {'annotations': {'kubernetes.io/change-cause': 'NOPE'}}
            data = func(name='test-pod', namespace='test', metadata=test_metadata)
            self.assertEqual(data.annotations, {'kubernetes.io/change-cause': 'NOPE'})

    def test_enforce_only_strings_dict(self):
        if False:
            print('Hello World!')
        func = getattr(kubernetes, '__enforce_only_strings_dict')
        data = {'unicode': 1, 2: 2}
        self.assertEqual({'unicode': '1', '2': '2'}, func(data))