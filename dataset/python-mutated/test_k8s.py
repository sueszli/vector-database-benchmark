"""
Unit Tests for the k8s execution module.
"""
import base64
import hashlib
import time
from subprocess import PIPE, Popen
import pytest
import salt.modules.k8s as k8s
import salt.utils.files
import salt.utils.json
from tests.support.unit import TestCase

@pytest.mark.skip_if_binaries_missing('kubectl')
class TestK8SNamespace(TestCase):
    maxDiff = None

    def test_get_namespaces(self):
        if False:
            i = 10
            return i + 15
        res = k8s.get_namespaces(apiserver_url='http://127.0.0.1:8080')
        a = len(res.get('items'))
        proc = Popen(['kubectl', 'get', 'namespaces', '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = len(kubectl_out.get('items'))
        self.assertEqual(a, b)

    def test_get_one_namespace(self):
        if False:
            for i in range(10):
                print('nop')
        res = k8s.get_namespaces('default', apiserver_url='http://127.0.0.1:8080')
        a = res.get('metadata', {}).get('name', 'a')
        proc = Popen(['kubectl', 'get', 'namespaces', 'default', '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('metadata', {}).get('name', 'b')
        self.assertEqual(a, b)

    def test_create_namespace(self):
        if False:
            while True:
                i = 10
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        nsname = hash.hexdigest()[:16]
        res = k8s.create_namespace(nsname, apiserver_url='http://127.0.0.1:8080')
        proc = Popen(['kubectl', 'get', 'namespaces', nsname, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        self.assertTrue(isinstance(kubectl_out, dict))

@pytest.mark.skip_if_binaries_missing('kubectl')
class TestK8SSecrets(TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            while True:
                i = 10
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        self.name = hash.hexdigest()[:16]
        data = {'testsecret': base64.encodestring('teststring')}
        self.request = {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': self.name, 'namespace': 'default'}, 'data': data}

    def test_get_secrets(self):
        if False:
            i = 10
            return i + 15
        res = k8s.get_secrets('default', apiserver_url='http://127.0.0.1:8080')
        a = len(res.get('items', []))
        proc = Popen(['kubectl', '--namespace=default', 'get', 'secrets', '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = len(kubectl_out.get('items', []))
        self.assertEqual(a, b)

    def test_get_one_secret(self):
        if False:
            return 10
        name = self.name
        filename = '/tmp/{}.json'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            salt.utils.json.dump(self.request, f)
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        res = k8s.get_secrets('default', name, apiserver_url='http://127.0.0.1:8080')
        a = res.get('metadata', {}).get('name', 'a')
        proc = Popen(['kubectl', '--namespace=default', 'get', 'secrets', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('metadata', {}).get('name', 'b')
        self.assertEqual(a, b)

    def test_get_decoded_secret(self):
        if False:
            return 10
        name = self.name
        filename = '/tmp/{}.json'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            salt.utils.json.dump(self.request, f)
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        res = k8s.get_secrets('default', name, apiserver_url='http://127.0.0.1:8080', decode=True)
        a = res.get('data', {}).get('testsecret')
        self.assertEqual(a, 'teststring')

    def test_create_secret(self):
        if False:
            i = 10
            return i + 15
        name = self.name
        names = []
        expected_data = {}
        for i in range(2):
            names.append('/tmp/{}-{}'.format(name, i))
            with salt.utils.files.fopen('/tmp/{}-{}'.format(name, i), 'w') as f:
                expected_data['{}-{}'.format(name, i)] = base64.b64encode('{}{}'.format(name, i))
                f.write(salt.utils.stringutils.to_str('{}{}'.format(name, i)))
        res = k8s.create_secret('default', name, names, apiserver_url='http://127.0.0.1:8080')
        proc = Popen(['kubectl', '--namespace=default', 'get', 'secrets', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('data', {})
        self.assertTrue(isinstance(kubectl_out, dict))
        self.assertEqual(expected_data, b)

    def test_update_secret(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.name
        filename = '/tmp/{}.json'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            salt.utils.json.dump(self.request, f)
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        expected_data = {}
        names = []
        for i in range(3):
            names.append('/tmp/{}-{}-updated'.format(name, i))
            with salt.utils.files.fopen('/tmp/{}-{}-updated'.format(name, i), 'w') as f:
                expected_data['{}-{}-updated'.format(name, i)] = base64.b64encode('{}{}-updated'.format(name, i))
                f.write('{}{}-updated'.format(name, i))
        res = k8s.update_secret('default', name, names, apiserver_url='http://127.0.0.1:8080')
        proc = Popen(['kubectl', '--namespace=default', 'get', 'secrets', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('data', {})
        self.assertTrue(isinstance(kubectl_out, dict))
        self.assertEqual(expected_data, b)

    def test_delete_secret(self):
        if False:
            print('Hello World!')
        name = self.name
        filename = '/tmp/{}.json'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            salt.utils.json.dump(self.request, f)
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        res = k8s.delete_secret('default', name, apiserver_url='http://127.0.0.1:8080')
        time.sleep(0.1)
        proc = Popen(['kubectl', '--namespace=default', 'get', 'secrets', name, '-o', 'json'], stdout=PIPE, stderr=PIPE)
        (kubectl_out, err) = proc.communicate()
        self.assertEqual('', kubectl_out)
        self.assertEqual('Error from server: secrets "{}" not found\n'.format(name), err)

@pytest.mark.skip_if_binaries_missing('kubectl')
class TestK8SResourceQuotas(TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            while True:
                i = 10
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        self.name = hash.hexdigest()[:16]

    def test_get_resource_quotas(self):
        if False:
            return 10
        name = self.name
        namespace = self.name
        create_namespace = Popen(['kubectl', 'create', 'namespace', namespace], stdout=PIPE)
        create_namespace = Popen(['kubectl', 'create', 'namespace', namespace], stdout=PIPE)
        request = '\napiVersion: v1\nkind: ResourceQuota\nmetadata:\n  name: {}\nspec:\n  hard:\n    cpu: "20"\n    memory: 1Gi\n    persistentvolumeclaims: "10"\n    pods: "10"\n    replicationcontrollers: "20"\n    resourcequotas: "1"\n    secrets: "10"\n    services: "5"\n'.format(name)
        filename = '/tmp/{}.yaml'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            f.write(salt.utils.stringutils.to_str(request))
        create = Popen(['kubectl', '--namespace={}'.format(namespace), 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.2)
        res = k8s.get_resource_quotas(namespace, apiserver_url='http://127.0.0.1:8080')
        a = len(res.get('items', []))
        proc = Popen(['kubectl', '--namespace={}'.format(namespace), 'get', 'quota', '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = len(kubectl_out.get('items', []))
        self.assertEqual(a, b)

    def test_get_one_resource_quota(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.name
        namespace = self.name
        create_namespace = Popen(['kubectl', 'create', 'namespace', namespace], stdout=PIPE)
        request = '\napiVersion: v1\nkind: ResourceQuota\nmetadata:\n  name: {}\nspec:\n  hard:\n    cpu: "20"\n    memory: 1Gi\n    persistentvolumeclaims: "10"\n    pods: "10"\n    replicationcontrollers: "20"\n    resourcequotas: "1"\n    secrets: "10"\n    services: "5"\n'.format(name)
        filename = '/tmp/{}.yaml'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            f.write(salt.utils.stringutils.to_str(request))
        create = Popen(['kubectl', '--namespace={}'.format(namespace), 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.2)
        res = k8s.get_resource_quotas(namespace, name, apiserver_url='http://127.0.0.1:8080')
        a = res.get('metadata', {}).get('name', 'a')
        proc = Popen(['kubectl', '--namespace={}'.format(namespace), 'get', 'quota', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('metadata', {}).get('name', 'b')
        self.assertEqual(a, b)

    def test_create_resource_quota(self):
        if False:
            print('Hello World!')
        name = self.name
        namespace = self.name
        create_namespace = Popen(['kubectl', 'create', 'namespace', namespace], stdout=PIPE)
        quota = {'cpu': '20', 'memory': '1Gi'}
        res = k8s.create_resource_quota(namespace, quota, name=name, apiserver_url='http://127.0.0.1:8080')
        proc = Popen(['kubectl', '--namespace={}'.format(namespace), 'get', 'quota', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        self.assertTrue(isinstance(kubectl_out, dict))

    def test_update_resource_quota(self):
        if False:
            while True:
                i = 10
        name = self.name
        namespace = self.name
        create_namespace = Popen(['kubectl', 'create', 'namespace', namespace], stdout=PIPE)
        request = '\napiVersion: v1\nkind: ResourceQuota\nmetadata:\n  name: {}\nspec:\n  hard:\n    cpu: "20"\n    memory: 1Gi\n    persistentvolumeclaims: "10"\n    pods: "10"\n    replicationcontrollers: "20"\n    resourcequotas: "1"\n    secrets: "10"\n    services: "5"\n'.format(name)
        filename = '/tmp/{}.yaml'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            f.write(salt.utils.stringutils.to_str(request))
        create = Popen(['kubectl', '--namespace={}'.format(namespace), 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.2)
        quota = {'cpu': '10', 'memory': '2Gi'}
        res = k8s.create_resource_quota(namespace, quota, name=name, apiserver_url='http://127.0.0.1:8080', update=True)
        proc = Popen(['kubectl', '--namespace={}'.format(namespace), 'get', 'quota', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        limit = kubectl_out.get('spec').get('hard').get('memory')
        self.assertEqual('2Gi', limit)

@pytest.mark.skip_if_binaries_missing('kubectl')
class TestK8SLimitRange(TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        hash = hashlib.sha1()
        hash.update(str(time.time()))
        self.name = hash.hexdigest()[:16]

    def test_create_limit_range(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.name
        limits = {'Container': {'defaultRequest': {'cpu': '100m'}}}
        res = k8s.create_limit_range('default', limits, name=name, apiserver_url='http://127.0.0.1:8080')
        proc = Popen(['kubectl', '--namespace=default', 'get', 'limits', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        self.assertTrue(isinstance(kubectl_out, dict))

    def test_update_limit_range(self):
        if False:
            i = 10
            return i + 15
        name = self.name
        request = '\napiVersion: v1\nkind: LimitRange\nmetadata:\n  name: {}\nspec:\n  limits:\n  - default:\n      cpu: 200m\n      memory: 512Mi\n    defaultRequest:\n      cpu: 100m\n      memory: 256Mi\n    type: Container\n'.format(name)
        limits = {'Container': {'defaultRequest': {'cpu': '100m'}}}
        filename = '/tmp/{}.yaml'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            f.write(salt.utils.stringutils.to_str(request))
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        res = k8s.create_limit_range('default', limits, name=name, apiserver_url='http://127.0.0.1:8080', update=True)
        proc = Popen(['kubectl', '--namespace=default', 'get', 'limits', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        limit = kubectl_out.get('spec').get('limits')[0].get('defaultRequest').get('cpu')
        self.assertEqual('100m', limit)

    def test_get_limit_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        res = k8s.get_limit_ranges('default', apiserver_url='http://127.0.0.1:8080')
        a = len(res.get('items', []))
        proc = Popen(['kubectl', '--namespace=default', 'get', 'limits', '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = len(kubectl_out.get('items', []))
        self.assertEqual(a, b)

    def test_get_one_limit_range(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.name
        request = '\napiVersion: v1\nkind: LimitRange\nmetadata:\n  name: {}\nspec:\n  limits:\n  - default:\n      cpu: 200m\n      memory: 512Mi\n    defaultRequest:\n      cpu: 100m\n      memory: 256Mi\n    type: Container\n'.format(name)
        filename = '/tmp/{}.yaml'.format(name)
        with salt.utils.files.fopen(filename, 'w') as f:
            f.write(salt.utils.stringutils.to_str(request))
        create = Popen(['kubectl', '--namespace=default', 'create', '-f', filename], stdout=PIPE)
        time.sleep(0.1)
        res = k8s.get_limit_ranges('default', name, apiserver_url='http://127.0.0.1:8080')
        a = res.get('metadata', {}).get('name', 'a')
        proc = Popen(['kubectl', '--namespace=default', 'get', 'limits', name, '-o', 'json'], stdout=PIPE)
        kubectl_out = salt.utils.json.loads(proc.communicate()[0])
        b = kubectl_out.get('metadata', {}).get('name', 'b')
        self.assertEqual(a, b)