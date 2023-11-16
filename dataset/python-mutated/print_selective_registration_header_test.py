"""Tests for print_selective_registration_header."""
import os
import sys
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.tools import selective_registration_header_lib
GRAPH_DEF_TXT = '\n  node: {\n    name: "node_1"\n    op: "Reshape"\n    input: [ "none", "none" ]\n    device: "/cpu:0"\n    attr: { key: "T" value: { type: DT_FLOAT } }\n  }\n  node: {\n    name: "node_2"\n    op: "MatMul"\n    input: [ "none", "none" ]\n    device: "/cpu:0"\n    attr: { key: "T" value: { type: DT_FLOAT } }\n    attr: { key: "transpose_a" value: { b: false } }\n    attr: { key: "transpose_b" value: { b: false } }\n  }\n  node: {\n    name: "node_3"\n    op: "MatMul"\n    input: [ "none", "none" ]\n    device: "/cpu:0"\n    attr: { key: "T" value: { type: DT_DOUBLE } }\n    attr: { key: "transpose_a" value: { b: false } }\n    attr: { key: "transpose_b" value: { b: false } }\n  }\n  library {\n    function {\n      node_def {\n        name: "node_6"\n        op: "Const"\n        attr: { key: "dtype" value: { type: DT_INT64 } }\n      }\n      node_def {\n        name: "node_7"\n        op: "Maximum"\n        input: "clip_by_value/Minimum:z:0"\n        input: "clip_by_value/y:output:0"\n        attr: { key: "T" value: { type: DT_INT64 } }\n        attr { key: "_output_shapes" value: { list: { shape: { dim: { size: -1 } dim: { size: 1 } } } } }\n      }\n    }\n  }\n'
GRAPH_DEF_TXT_2 = '\n  node: {\n    name: "node_4"\n    op: "BiasAdd"\n    input: [ "none", "none" ]\n    device: "/cpu:0"\n    attr: { key: "T" value: { type: DT_FLOAT } }\n  }\n  node: {\n    name: "node_5"\n    op: "AccumulateNV2"\n    attr: { key: "T" value: { type: DT_INT32 } }\n    attr: { key  : "N" value: { i: 3 } }\n  }\n\n'

class PrintOpFilegroupTest(test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        (_, self.script_name) = os.path.split(sys.argv[0])

    def WriteGraphFiles(self, graphs):
        if False:
            for i in range(10):
                print('nop')
        fnames = []
        for (i, graph) in enumerate(graphs):
            fname = os.path.join(self.get_temp_dir(), 'graph%s.pb' % i)
            with gfile.GFile(fname, 'wb') as f:
                f.write(graph.SerializeToString())
            fnames.append(fname)
        return fnames

    def WriteTextFile(self, content):
        if False:
            for i in range(10):
                print('nop')
        fname = os.path.join(self.get_temp_dir(), 'text.txt')
        with gfile.GFile(fname, 'w') as f:
            f.write(content)
        return [fname]

    def testGetOps(self):
        if False:
            while True:
                i = 10
        default_ops = 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp'
        graphs = [text_format.Parse(d, graph_pb2.GraphDef()) for d in [GRAPH_DEF_TXT, GRAPH_DEF_TXT_2]]
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('rawproto', self.WriteGraphFiles(graphs), default_ops)
        matmul_prefix = 'Batch'
        self.assertListEqual([('AccumulateNV2', None), ('BiasAdd', 'BiasOp<CPUDevice, float>'), ('Const', 'ConstantOp'), ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, double, double, double, true>'), ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, float, float, float, true>'), ('Maximum', 'BinaryOp<CPUDevice, functor::maximum<int64_t>>'), ('NoOp', 'NoOp'), ('Reshape', 'ReshapeOp'), ('_Recv', 'RecvOp'), ('_Send', 'SendOp')], ops_and_kernels)
        graphs[0].node[0].ClearField('device')
        graphs[0].node[2].ClearField('device')
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('rawproto', self.WriteGraphFiles(graphs), default_ops)
        self.assertListEqual([('AccumulateNV2', None), ('BiasAdd', 'BiasOp<CPUDevice, float>'), ('Const', 'ConstantOp'), ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, double, double, double, true>'), ('MatMul', matmul_prefix + 'MatMulOp<CPUDevice, float, float, float, true>'), ('Maximum', 'BinaryOp<CPUDevice, functor::maximum<int64_t>>'), ('NoOp', 'NoOp'), ('Reshape', 'ReshapeOp'), ('_Recv', 'RecvOp'), ('_Send', 'SendOp')], ops_and_kernels)

    def testGetOpsFromList(self):
        if False:
            print('Hello World!')
        default_ops = ''
        ops_list = '[["Add", "BinaryOp<CPUDevice, functor::add<float>>"],\n        ["Softplus", "SoftplusOp<CPUDevice, float>"]]'
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list), default_ops)
        self.assertListEqual([('Add', 'BinaryOp<CPUDevice, functor::add<float>>'), ('Softplus', 'SoftplusOp<CPUDevice, float>')], ops_and_kernels)
        ops_list = '[["Softplus", "SoftplusOp<CPUDevice, float>"]]'
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list), default_ops)
        self.assertListEqual([('Softplus', 'SoftplusOp<CPUDevice, float>')], ops_and_kernels)
        ops_list = '[["Add", "BinaryOp<CPUDevice, functor::add<float>>"],\n        ["Add", "BinaryOp<CPUDevice, functor::add<float>>"]]'
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list), default_ops)
        self.assertListEqual([('Add', 'BinaryOp<CPUDevice, functor::add<float>>')], ops_and_kernels)
        ops_list = '[["Softplus", ""]]'
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list), default_ops)
        self.assertListEqual([('Softplus', None)], ops_and_kernels)
        ops_list = '[["Softplus", "SoftplusOp<CPUDevice, float>"]]'
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list) + self.WriteTextFile(ops_list), default_ops)
        self.assertListEqual([('Softplus', 'SoftplusOp<CPUDevice, float>')], ops_and_kernels)
        ops_list = ''
        with self.assertRaises(Exception):
            ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('ops_list', self.WriteTextFile(ops_list), default_ops)

    def testAll(self):
        if False:
            print('Hello World!')
        default_ops = 'all'
        graphs = [text_format.Parse(d, graph_pb2.GraphDef()) for d in [GRAPH_DEF_TXT, GRAPH_DEF_TXT_2]]
        ops_and_kernels = selective_registration_header_lib.get_ops_and_kernels('rawproto', self.WriteGraphFiles(graphs), default_ops)
        header = selective_registration_header_lib.get_header_from_ops_and_kernels(ops_and_kernels, include_all_ops_and_kernels=True)
        self.assertListEqual(['// This file was autogenerated by %s' % self.script_name, '#ifndef OPS_TO_REGISTER', '#define OPS_TO_REGISTER', '#define SHOULD_REGISTER_OP(op) true', '#define SHOULD_REGISTER_OP_KERNEL(clz) true', '#define SHOULD_REGISTER_OP_GRADIENT true', '#endif'], header.split('\n'))
        self.assertListEqual(header.split('\n'), selective_registration_header_lib.get_header(self.WriteGraphFiles(graphs), 'rawproto', default_ops).split('\n'))

    def testGetSelectiveHeader(self):
        if False:
            return 10
        default_ops = ''
        graphs = [text_format.Parse(GRAPH_DEF_TXT_2, graph_pb2.GraphDef())]
        expected = '// This file was autogenerated by %s\n#ifndef OPS_TO_REGISTER\n#define OPS_TO_REGISTER\n\n    namespace {\n      constexpr const char* skip(const char* x) {\n        return (*x) ? (*x == \' \' ? skip(x + 1) : x) : x;\n      }\n\n      constexpr bool isequal(const char* x, const char* y) {\n        return (*skip(x) && *skip(y))\n                   ? (*skip(x) == *skip(y) && isequal(skip(x) + 1, skip(y) + 1))\n                   : (!*skip(x) && !*skip(y));\n      }\n\n      template<int N>\n      struct find_in {\n        static constexpr bool f(const char* x, const char* const y[N]) {\n          return isequal(x, y[0]) || find_in<N - 1>::f(x, y + 1);\n        }\n      };\n\n      template<>\n      struct find_in<0> {\n        static constexpr bool f(const char* x, const char* const y[]) {\n          return false;\n        }\n      };\n    }  // end namespace\n    constexpr const char* kNecessaryOpKernelClasses[] = {\n"BiasOp<CPUDevice, float>",\n};\n#define SHOULD_REGISTER_OP_KERNEL(clz) (find_in<sizeof(kNecessaryOpKernelClasses) / sizeof(*kNecessaryOpKernelClasses)>::f(clz, kNecessaryOpKernelClasses))\n\nconstexpr inline bool ShouldRegisterOp(const char op[]) {\n  return false\n     || isequal(op, "AccumulateNV2")\n     || isequal(op, "BiasAdd")\n  ;\n}\n#define SHOULD_REGISTER_OP(op) ShouldRegisterOp(op)\n\n#define SHOULD_REGISTER_OP_GRADIENT false\n#endif' % self.script_name
        header = selective_registration_header_lib.get_header(self.WriteGraphFiles(graphs), 'rawproto', default_ops)
        print(header)
        self.assertListEqual(expected.split('\n'), header.split('\n'))
if __name__ == '__main__':
    test.main()