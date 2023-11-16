import base64
import copy
import os
import sys
import textwrap
from io import StringIO
from unittest import mock
from unittest.case import SkipTest
import yaml
from twisted.internet import defer
from twisted.python import runtime
from twisted.trial import unittest
from buildbot.process.properties import Interpolate
from buildbot.test.fake import fakemaster
from buildbot.test.fake import httpclientservice as fakehttp
from buildbot.test.fake import kube as fakekube
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import config
from buildbot.util import kubeclientservice

class MockFileBase:
    file_mock_config = {}

    def setUp(self):
        if False:
            while True:
                i = 10
        self.patcher = mock.patch('buildbot.util.kubeclientservice.open', self.mock_open)
        self.patcher.start()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.patcher.stop()

    def mock_open(self, filename, mode=None, encoding='UTF-8'):
        if False:
            return 10
        filename_type = os.path.basename(filename)
        file_value = self.file_mock_config[filename_type]
        mock_open = mock.Mock(__enter__=mock.Mock(return_value=StringIO(file_value)), __exit__=mock.Mock())
        return mock_open

class KubeClientServiceTestClusterConfig(MockFileBase, config.ConfigErrorsMixin, unittest.TestCase):
    file_mock_config = {'token': 'BASE64_TOKEN', 'namespace': 'buildbot_namespace'}

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.patch(kubeclientservice.os, 'environ', {'KUBERNETES_PORT': 'tcp://foo'})

    def patchExist(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.patch(kubeclientservice.os.path, 'exists', lambda x: val)

    def test_not_exists(self):
        if False:
            while True:
                i = 10
        self.patchExist(False)
        with self.assertRaisesConfigError('kube_dir not found:'):
            kubeclientservice.KubeInClusterConfigLoader()

    @defer.inlineCallbacks
    def test_basic(self):
        if False:
            print('Hello World!')
        self.patchExist(True)
        config = kubeclientservice.KubeInClusterConfigLoader()
        yield config.startService()
        self.assertEqual(config.getConfig(), {'headers': {'Authorization': 'Bearer BASE64_TOKEN'}, 'master_url': 'https://foo', 'namespace': 'buildbot_namespace', 'verify': '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'})
KUBE_CTL_PROXY_FAKE = '\nimport time\nimport sys\n\nprint("Starting to serve on 127.0.0.1:" + sys.argv[2])\nsys.stdout.flush()\ntime.sleep(1000)\n'
KUBE_CTL_PROXY_FAKE_ERROR = '\nimport time\nimport sys\n\nprint("Issue with the config!", file=sys.stderr)\nsys.stderr.flush()\nsys.exit(1)\n'

class KubeClientServiceTestKubeHardcodedConfig(config.ConfigErrorsMixin, unittest.TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default')
        self.assertEqual(config.getConfig(), {'master_url': 'http://localhost:8001', 'namespace': 'default', 'headers': {}})

    @defer.inlineCallbacks
    def test_verify_is_forwarded_to_keywords(self):
        if False:
            i = 10
            return i + 15
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default', verify='/path/to/pem')
        service = kubeclientservice.KubeClientService(config)
        (_, kwargs) = (yield service._prepareRequest('/test', {}))
        self.assertEqual('/path/to/pem', kwargs['verify'])

    @defer.inlineCallbacks
    def test_verify_headers_are_passed_to_the_query(self):
        if False:
            for i in range(10):
                print('nop')
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default', verify='/path/to/pem', headers={'Test': '10'})
        service = kubeclientservice.KubeClientService(config)
        (_, kwargs) = (yield service._prepareRequest('/test', {}))
        self.assertEqual({'Test': '10'}, kwargs['headers'])

    def test_the_configuration_parent_is_set_to_the_service(self):
        if False:
            while True:
                i = 10
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001')
        service = kubeclientservice.KubeClientService(config)
        self.assertEqual(service, self.config.parent)

    def test_cannot_pass_both_bearer_and_basic_auth(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default', verify='/path/to/pem', basicAuth='Bla', bearerToken='Bla')

    @defer.inlineCallbacks
    def test_verify_bearerToken_is_expanded(self):
        if False:
            for i in range(10):
                print('nop')
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default', verify='/path/to/pem', bearerToken=Interpolate('%(kw:test)s', test=10))
        service = kubeclientservice.KubeClientService(config)
        (_, kwargs) = (yield service._prepareRequest('/test', {}))
        self.assertEqual('Bearer 10', kwargs['headers']['Authorization'])

    @defer.inlineCallbacks
    def test_verify_basicAuth_is_expanded(self):
        if False:
            for i in range(10):
                print('nop')
        self.config = config = kubeclientservice.KubeHardcodedConfig(master_url='http://localhost:8001', namespace='default', verify='/path/to/pem', basicAuth={'user': 'name', 'password': Interpolate('%(kw:test)s', test=10)})
        service = kubeclientservice.KubeClientService(config)
        (_, kwargs) = (yield service._prepareRequest('/test', {}))
        expected = f"Basic {base64.b64encode('name:10'.encode('utf-8'))}"
        self.assertEqual(expected, kwargs['headers']['Authorization'])

class KubeClientServiceTestKubeCtlProxyConfig(config.ConfigErrorsMixin, unittest.TestCase):

    def patchProxyCmd(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        if runtime.platformType != 'posix':
            self.config = None
            raise SkipTest('only posix platform is supported by this test')
        self.patch(kubeclientservice.KubeCtlProxyConfigLoader, 'kube_ctl_proxy_cmd', [sys.executable, '-c', cmd])

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.config is not None:
            return self.config.stopService()
        return None

    @defer.inlineCallbacks
    def test_basic(self):
        if False:
            print('Hello World!')
        self.patchProxyCmd(KUBE_CTL_PROXY_FAKE)
        self.config = config = kubeclientservice.KubeCtlProxyConfigLoader()
        yield config.startService()
        self.assertEqual(config.getConfig(), {'master_url': 'http://localhost:8001', 'namespace': 'default'})

    @defer.inlineCallbacks
    def test_config_args(self):
        if False:
            while True:
                i = 10
        self.patchProxyCmd(KUBE_CTL_PROXY_FAKE)
        self.config = config = kubeclientservice.KubeCtlProxyConfigLoader(proxy_port=8002, namespace='system')
        yield config.startService()
        self.assertEqual(config.kube_proxy_output, b'Starting to serve on 127.0.0.1:8002')
        self.assertEqual(config.getConfig(), {'master_url': 'http://localhost:8002', 'namespace': 'system'})
        yield config.stopService()

    @defer.inlineCallbacks
    def test_config_with_error(self):
        if False:
            i = 10
            return i + 15
        self.patchProxyCmd(KUBE_CTL_PROXY_FAKE_ERROR)
        self.config = config = kubeclientservice.KubeCtlProxyConfigLoader()
        with self.assertRaises(RuntimeError):
            yield config.startService()

class RealKubeClientServiceTest(TestReactorMixin, unittest.TestCase):
    timeout = 200
    POD_SPEC = yaml.safe_load(textwrap.dedent('\n    apiVersion: v1\n    kind: Pod\n    metadata:\n        name: pod-example\n    spec:\n        containers:\n        - name: alpine\n          image: alpine\n          command: ["sleep"]\n          args: ["100"]\n    '))

    def createKube(self):
        if False:
            while True:
                i = 10
        if 'TEST_KUBERNETES' not in os.environ:
            raise SkipTest('kubernetes integration tests only run when environment variable TEST_KUBERNETES is set')
        self.kube = kubeclientservice.KubeClientService(kubeclientservice.KubeCtlProxyConfigLoader())

    def expect(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self)
        self.createKube()
        yield self.kube.setServiceParent(self.master)
        yield self.master.startService()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.master.stopService()
    kube = None

    @defer.inlineCallbacks
    def test_create_and_delete_pod(self):
        if False:
            for i in range(10):
                print('nop')
        content = {'kind': 'Pod', 'metadata': {'name': 'pod-example'}}
        self.expect(method='post', ep='/api/v1/namespaces/default/pods', params=None, data=None, json={'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': 'pod-example'}, 'spec': {'containers': [{'name': 'alpine', 'image': 'alpine', 'command': ['sleep'], 'args': ['100']}]}}, content_json=content)
        res = (yield self.kube.createPod(self.kube.namespace, self.POD_SPEC))
        self.assertEqual(res['kind'], 'Pod')
        self.assertEqual(res['metadata']['name'], 'pod-example')
        self.assertNotIn('deletionTimestamp', res['metadata'])
        content['metadata']['deletionTimestamp'] = 'now'
        self.expect(method='delete', ep='/api/v1/namespaces/default/pods/pod-example', params={'graceperiod': 0}, data=None, json=None, code=200, content_json=content)
        res = (yield self.kube.deletePod(self.kube.namespace, 'pod-example'))
        self.assertEqual(res['kind'], 'Pod')
        self.assertIn('deletionTimestamp', res['metadata'])
        self.expect(method='get', ep='/api/v1/namespaces/default/pods/pod-example/status', params=None, data=None, json=None, code=200, content_json=content)
        content = {'kind': 'Status', 'reason': 'NotFound'}
        self.expect(method='get', ep='/api/v1/namespaces/default/pods/pod-example/status', params=None, data=None, json=None, code=404, content_json=content)
        res = (yield self.kube.waitForPodDeletion(self.kube.namespace, 'pod-example', timeout=200))
        self.assertEqual(res['kind'], 'Status')
        self.assertEqual(res['reason'], 'NotFound')

    @defer.inlineCallbacks
    def test_create_bad_spec(self):
        if False:
            return 10
        spec = copy.deepcopy(self.POD_SPEC)
        del spec['metadata']
        content = {'kind': 'Status', 'reason': 'MissingName', 'message': 'need name'}
        self.expect(method='post', ep='/api/v1/namespaces/default/pods', params=None, data=None, json={'apiVersion': 'v1', 'kind': 'Pod', 'spec': {'containers': [{'name': 'alpine', 'image': 'alpine', 'command': ['sleep'], 'args': ['100']}]}}, code=400, content_json=content)
        with self.assertRaises(kubeclientservice.KubeError):
            yield self.kube.createPod(self.kube.namespace, spec)

    @defer.inlineCallbacks
    def test_delete_not_existing(self):
        if False:
            for i in range(10):
                print('nop')
        content = {'kind': 'Status', 'reason': 'NotFound', 'message': 'no container by that name'}
        self.expect(method='delete', ep='/api/v1/namespaces/default/pods/pod-example', params={'graceperiod': 0}, data=None, json=None, code=404, content_json=content)
        with self.assertRaises(kubeclientservice.KubeError):
            yield self.kube.deletePod(self.kube.namespace, 'pod-example')

    @defer.inlineCallbacks
    def test_wait_for_delete_not_deleting(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.kube.createPod(self.kube.namespace, self.POD_SPEC)
        with self.assertRaises(TimeoutError):
            yield self.kube.waitForPodDeletion(self.kube.namespace, 'pod-example', timeout=2)
        res = (yield self.kube.deletePod(self.kube.namespace, 'pod-example'))
        self.assertEqual(res['kind'], 'Pod')
        self.assertIn('deletionTimestamp', res['metadata'])
        yield self.kube.waitForPodDeletion(self.kube.namespace, 'pod-example', timeout=100)

class FakeKubeClientServiceTest(RealKubeClientServiceTest):

    def createKube(self):
        if False:
            i = 10
            return i + 15
        self.kube = fakekube.KubeClientService(kubeclientservice.KubeHardcodedConfig(master_url='http://m'))

class PatchedKubeClientServiceTest(RealKubeClientServiceTest):

    def createKube(self):
        if False:
            return 10
        self.kube = kubeclientservice.KubeClientService(kubeclientservice.KubeHardcodedConfig(master_url='http://m'))
        self.http = fakehttp.HTTPClientService('http://m')
        self.kube.get = self.http.get
        self.kube.post = self.http.post
        self.kube.put = self.http.put
        self.kube.delete = self.http.delete

    def expect(self, *args, **kwargs):
        if False:
            return 10
        return self.http.expect(*args, **kwargs)

    def test_wait_for_delete_not_deleting(self):
        if False:
            print('Hello World!')
        pass