from cimodel.lib.conf_tree import ConfigNode
CONFIG_TREE_DATA = []

def get_major_pyver(dotted_version):
    if False:
        print('Hello World!')
    parts = dotted_version.split('.')
    return 'py' + parts[0]

class TreeConfigNode(ConfigNode):

    def __init__(self, parent, node_name, subtree):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, self.modify_label(node_name))
        self.subtree = subtree
        self.init2(node_name)

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return label

    def init2(self, node_name):
        if False:
            return 10
        pass

    def get_children(self):
        if False:
            while True:
                i = 10
        return [self.child_constructor()(self, k, v) for (k, v) in self.subtree]

class TopLevelNode(TreeConfigNode):

    def __init__(self, node_name, subtree):
        if False:
            return 10
        super().__init__(None, node_name, subtree)

    def child_constructor(self):
        if False:
            print('Hello World!')
        return DistroConfigNode

class DistroConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            for i in range(10):
                print('nop')
        self.props['distro_name'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        distro = self.find_prop('distro_name')
        next_nodes = {'xenial': XenialCompilerConfigNode, 'bionic': BionicCompilerConfigNode}
        return next_nodes[distro]

class PyVerConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['pyver'] = node_name
        self.props['abbreviated_pyver'] = get_major_pyver(node_name)
        if node_name == '3.9':
            self.props['abbreviated_pyver'] = 'py3.9'

    def child_constructor(self):
        if False:
            while True:
                i = 10
        return ExperimentalFeatureConfigNode

class ExperimentalFeatureConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            return 10
        self.props['experimental_feature'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        experimental_feature = self.find_prop('experimental_feature')
        next_nodes = {'asan': AsanConfigNode, 'xla': XlaConfigNode, 'mps': MPSConfigNode, 'vulkan': VulkanConfigNode, 'parallel_tbb': ParallelTBBConfigNode, 'crossref': CrossRefConfigNode, 'dynamo': DynamoConfigNode, 'parallel_native': ParallelNativeConfigNode, 'onnx': ONNXConfigNode, 'libtorch': LibTorchConfigNode, 'important': ImportantConfigNode, 'build_only': BuildOnlyConfigNode, 'shard_test': ShardTestConfigNode, 'cuda_gcc_override': CudaGccOverrideConfigNode, 'pure_torch': PureTorchConfigNode, 'slow_gradcheck': SlowGradcheckConfigNode}
        return next_nodes[experimental_feature]

class SlowGradcheckConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            i = 10
            return i + 15
        self.props['is_slow_gradcheck'] = True

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return ExperimentalFeatureConfigNode

class PureTorchConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            print('Hello World!')
        return 'PURE_TORCH=' + str(label)

    def init2(self, node_name):
        if False:
            for i in range(10):
                print('nop')
        self.props['is_pure_torch'] = node_name

    def child_constructor(self):
        if False:
            return 10
        return ImportantConfigNode

class XlaConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            return 10
        return 'XLA=' + str(label)

    def init2(self, node_name):
        if False:
            return 10
        self.props['is_xla'] = node_name

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return ImportantConfigNode

class MPSConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            print('Hello World!')
        return 'MPS=' + str(label)

    def init2(self, node_name):
        if False:
            while True:
                i = 10
        self.props['is_mps'] = node_name

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return ImportantConfigNode

class AsanConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return 'Asan=' + str(label)

    def init2(self, node_name):
        if False:
            return 10
        self.props['is_asan'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        return ExperimentalFeatureConfigNode

class ONNXConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            return 10
        return 'Onnx=' + str(label)

    def init2(self, node_name):
        if False:
            while True:
                i = 10
        self.props['is_onnx'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        return ImportantConfigNode

class VulkanConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return 'Vulkan=' + str(label)

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['is_vulkan'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        return ImportantConfigNode

class ParallelTBBConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return 'PARALLELTBB=' + str(label)

    def init2(self, node_name):
        if False:
            for i in range(10):
                print('nop')
        self.props['parallel_backend'] = 'paralleltbb'

    def child_constructor(self):
        if False:
            return 10
        return ImportantConfigNode

class CrossRefConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['is_crossref'] = node_name

    def child_constructor(self):
        if False:
            while True:
                i = 10
        return ImportantConfigNode

class DynamoConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            i = 10
            return i + 15
        self.props['is_dynamo'] = node_name

    def child_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        return ImportantConfigNode

class ParallelNativeConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return 'PARALLELNATIVE=' + str(label)

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['parallel_backend'] = 'parallelnative'

    def child_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        return ImportantConfigNode

class LibTorchConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            i = 10
            return i + 15
        return 'BUILD_TEST_LIBTORCH=' + str(label)

    def init2(self, node_name):
        if False:
            return 10
        self.props['is_libtorch'] = node_name

    def child_constructor(self):
        if False:
            while True:
                i = 10
        return ExperimentalFeatureConfigNode

class CudaGccOverrideConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            i = 10
            return i + 15
        self.props['cuda_gcc_override'] = node_name

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return ExperimentalFeatureConfigNode

class BuildOnlyConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            while True:
                i = 10
        self.props['build_only'] = node_name

    def child_constructor(self):
        if False:
            while True:
                i = 10
        return ExperimentalFeatureConfigNode

class ShardTestConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['shard_test'] = node_name

    def child_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        return ImportantConfigNode

class ImportantConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        return 'IMPORTANT=' + str(label)

    def init2(self, node_name):
        if False:
            i = 10
            return i + 15
        self.props['is_important'] = node_name

    def get_children(self):
        if False:
            return 10
        return []

class XenialCompilerConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            print('Hello World!')
        return label or '<unspecified>'

    def init2(self, node_name):
        if False:
            print('Hello World!')
        self.props['compiler_name'] = node_name

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return XenialCompilerVersionConfigNode if self.props['compiler_name'] else PyVerConfigNode

class BionicCompilerConfigNode(TreeConfigNode):

    def modify_label(self, label):
        if False:
            while True:
                i = 10
        return label or '<unspecified>'

    def init2(self, node_name):
        if False:
            i = 10
            return i + 15
        self.props['compiler_name'] = node_name

    def child_constructor(self):
        if False:
            return 10
        return BionicCompilerVersionConfigNode if self.props['compiler_name'] else PyVerConfigNode

class XenialCompilerVersionConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            while True:
                i = 10
        self.props['compiler_version'] = node_name

    def child_constructor(self):
        if False:
            print('Hello World!')
        return PyVerConfigNode

class BionicCompilerVersionConfigNode(TreeConfigNode):

    def init2(self, node_name):
        if False:
            for i in range(10):
                print('nop')
        self.props['compiler_version'] = node_name

    def child_constructor(self):
        if False:
            i = 10
            return i + 15
        return PyVerConfigNode