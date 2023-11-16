import unittest
from prometheus_client import CollectorRegistry, PlatformCollector

class TestPlatformCollector(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.registry = CollectorRegistry()
        self.platform = _MockPlatform()

    def test_python_info(self):
        if False:
            return 10
        PlatformCollector(registry=self.registry, platform=self.platform)
        self.assertLabels('python_info', {'version': 'python_version', 'implementation': 'python_implementation', 'major': 'pvt_major', 'minor': 'pvt_minor', 'patchlevel': 'pvt_patchlevel'})

    def test_system_info_java(self):
        if False:
            print('Hello World!')
        self.platform._system = 'Java'
        PlatformCollector(registry=self.registry, platform=self.platform)
        self.assertLabels('python_info', {'version': 'python_version', 'implementation': 'python_implementation', 'major': 'pvt_major', 'minor': 'pvt_minor', 'patchlevel': 'pvt_patchlevel', 'jvm_version': 'jv_release', 'jvm_release': 'vm_release', 'jvm_vendor': 'vm_vendor', 'jvm_name': 'vm_name'})

    def assertLabels(self, name, labels):
        if False:
            return 10
        for metric in self.registry.collect():
            for s in metric.samples:
                if s.name == name:
                    assert s.labels == labels
                    return
        assert False

class _MockPlatform:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._system = 'system'

    def python_version_tuple(self):
        if False:
            while True:
                i = 10
        return ('pvt_major', 'pvt_minor', 'pvt_patchlevel')

    def python_version(self):
        if False:
            for i in range(10):
                print('nop')
        return 'python_version'

    def python_implementation(self):
        if False:
            return 10
        return 'python_implementation'

    def system(self):
        if False:
            while True:
                i = 10
        return self._system

    def java_ver(self):
        if False:
            i = 10
            return i + 15
        return ('jv_release', 'jv_vendor', ('vm_name', 'vm_release', 'vm_vendor'), ('os_name', 'os_version', 'os_arch'))