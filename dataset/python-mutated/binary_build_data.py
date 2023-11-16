"""
This module models the tree of configuration variants
for "smoketest" builds.

Each subclass of ConfigNode represents a layer of the configuration hierarchy.
These tree nodes encapsulate the logic for whether a branch of the hierarchy
should be "pruned".
"""
from collections import OrderedDict
import cimodel.data.dimensions as dimensions
from cimodel.lib.conf_tree import ConfigNode
LINKING_DIMENSIONS = ['shared', 'static']
DEPS_INCLUSION_DIMENSIONS = ['with-deps', 'without-deps']

def get_processor_arch_name(gpu_version):
    if False:
        i = 10
        return i + 15
    return 'cpu' if not gpu_version else 'cu' + gpu_version.strip('cuda') if gpu_version.startswith('cuda') else gpu_version
CONFIG_TREE_DATA = OrderedDict()
LINUX_GCC_CONFIG_VARIANTS = OrderedDict(manywheel=['devtoolset7'], conda=['devtoolset7'], libtorch=['devtoolset7', 'gcc5.4_cxx11-abi'])
WINDOWS_LIBTORCH_CONFIG_VARIANTS = ['debug', 'release']

class TopLevelNode(ConfigNode):

    def __init__(self, node_name, config_tree_data, smoke):
        if False:
            return 10
        super().__init__(None, node_name)
        self.config_tree_data = config_tree_data
        self.props['smoke'] = smoke

    def get_children(self):
        if False:
            while True:
                i = 10
        return [OSConfigNode(self, x, c, p) for (x, (c, p)) in self.config_tree_data.items()]

class OSConfigNode(ConfigNode):

    def __init__(self, parent, os_name, gpu_versions, py_tree):
        if False:
            print('Hello World!')
        super().__init__(parent, os_name)
        self.py_tree = py_tree
        self.props['os_name'] = os_name
        self.props['gpu_versions'] = gpu_versions

    def get_children(self):
        if False:
            print('Hello World!')
        return [PackageFormatConfigNode(self, k, v) for (k, v) in self.py_tree.items()]

class PackageFormatConfigNode(ConfigNode):

    def __init__(self, parent, package_format, python_versions):
        if False:
            return 10
        super().__init__(parent, package_format)
        self.props['python_versions'] = python_versions
        self.props['package_format'] = package_format

    def get_children(self):
        if False:
            return 10
        if self.find_prop('os_name') == 'linux':
            return [LinuxGccConfigNode(self, v) for v in LINUX_GCC_CONFIG_VARIANTS[self.find_prop('package_format')]]
        elif self.find_prop('os_name') == 'windows' and self.find_prop('package_format') == 'libtorch':
            return [WindowsLibtorchConfigNode(self, v) for v in WINDOWS_LIBTORCH_CONFIG_VARIANTS]
        else:
            return [ArchConfigNode(self, v) for v in self.find_prop('gpu_versions')]

class LinuxGccConfigNode(ConfigNode):

    def __init__(self, parent, gcc_config_variant):
        if False:
            print('Hello World!')
        super().__init__(parent, 'GCC_CONFIG_VARIANT=' + str(gcc_config_variant))
        self.props['gcc_config_variant'] = gcc_config_variant

    def get_children(self):
        if False:
            i = 10
            return i + 15
        gpu_versions = self.find_prop('gpu_versions')
        if self.find_prop('gcc_config_variant') == 'devtoolset7':
            gpu_versions = filter(lambda x: x != 'cuda_90', gpu_versions)
        if self.find_prop('package_format') == 'conda':
            gpu_versions = filter(lambda x: x not in dimensions.ROCM_VERSION_LABELS, gpu_versions)
        if self.find_prop('package_format') == 'libtorch':
            gpu_versions = filter(lambda x: x not in dimensions.ROCM_VERSION_LABELS, gpu_versions)
        return [ArchConfigNode(self, v) for v in gpu_versions]

class WindowsLibtorchConfigNode(ConfigNode):

    def __init__(self, parent, libtorch_config_variant):
        if False:
            return 10
        super().__init__(parent, 'LIBTORCH_CONFIG_VARIANT=' + str(libtorch_config_variant))
        self.props['libtorch_config_variant'] = libtorch_config_variant

    def get_children(self):
        if False:
            for i in range(10):
                print('nop')
        return [ArchConfigNode(self, v) for v in self.find_prop('gpu_versions')]

class ArchConfigNode(ConfigNode):

    def __init__(self, parent, gpu):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, get_processor_arch_name(gpu))
        self.props['gpu'] = gpu

    def get_children(self):
        if False:
            return 10
        return [PyVersionConfigNode(self, v) for v in self.find_prop('python_versions')]

class PyVersionConfigNode(ConfigNode):

    def __init__(self, parent, pyver):
        if False:
            print('Hello World!')
        super().__init__(parent, pyver)
        self.props['pyver'] = pyver

    def get_children(self):
        if False:
            while True:
                i = 10
        package_format = self.find_prop('package_format')
        os_name = self.find_prop('os_name')
        has_libtorch_variants = package_format == 'libtorch' and os_name == 'linux'
        linking_variants = LINKING_DIMENSIONS if has_libtorch_variants else []
        return [LinkingVariantConfigNode(self, v) for v in linking_variants]

class LinkingVariantConfigNode(ConfigNode):

    def __init__(self, parent, linking_variant):
        if False:
            i = 10
            return i + 15
        super().__init__(parent, linking_variant)

    def get_children(self):
        if False:
            while True:
                i = 10
        return [DependencyInclusionConfigNode(self, v) for v in DEPS_INCLUSION_DIMENSIONS]

class DependencyInclusionConfigNode(ConfigNode):

    def __init__(self, parent, deps_variant):
        if False:
            return 10
        super().__init__(parent, deps_variant)
        self.props['libtorch_variant'] = '-'.join([self.parent.get_label(), self.get_label()])