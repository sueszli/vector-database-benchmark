import os
import tempfile
import unittest
from typing import Dict
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader
from torchgen.gen_executorch import ComputeCodegenUnboxedKernels, gen_functions_declarations, parse_yaml_files, translate_native_yaml
from torchgen.model import BackendIndex, BackendMetadata, DispatchKey, Location, NativeFunction, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
TEST_YAML = '\n- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)\n  device_check: NoCheck   # TensorIterator\n  structured: True\n  structured_inherits: TensorIteratorBase\n  ufunc_inner_loop:\n    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)\n    ScalarOnly: add (Bool)\n  dispatch:\n    SparseCPU: add_out_sparse_cpu\n    SparseCUDA: add_out_sparse_cuda\n    SparseCsrCPU: add_out_sparse_csr_cpu\n    SparseCsrCUDA: add_out_sparse_csr_cuda\n    MkldnnCPU: mkldnn_add_out\n    MPS: add_out_mps\n\n- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor\n  device_check: NoCheck   # TensorIterator\n  structured_delegate: add.out\n  variants: function, method\n  dispatch:\n    SparseCPU, SparseCUDA: add_sparse\n    SparseCsrCPU, SparseCsrCUDA: add_sparse_csr\n    MkldnnCPU: mkldnn_add\n    ZeroTensor: add_zerotensor\n    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor\n  tags: core\n\n- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)\n  device_check: NoCheck   # TensorIterator\n  structured: True\n  structured_inherits: TensorIteratorBase\n  dispatch:\n    CPU, CUDA: mul_out\n    MPS: mul_out_mps\n    SparseCPU: mul_out_sparse_cpu\n    SparseCUDA: mul_out_sparse_cuda\n    SparseCsrCPU, SparseCsrCUDA: mul_out_sparse_csr\n    MkldnnCPU: mkldnn_mul_out\n\n- func: mul.Tensor(Tensor self, Tensor other) -> Tensor\n  device_check: NoCheck   # TensorIterator\n  structured_delegate: mul.out\n  variants: function, method\n  dispatch:\n    SparseCPU, SparseCUDA: mul_sparse\n    SparseCsrCPU, SparseCsrCUDA: mul_sparse_csr\n    MkldnnCPU: mkldnn_mul\n    ZeroTensor: mul_zerotensor\n    NestedTensorCPU, NestedTensorCUDA: NestedTensor_mul_Tensor\n  tags: core\n\n'
TEST_KERNEL_YAML = '\n- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)\n  device_check: NoCheck   # TensorIterator\n  structured: True\n  structured_inherits: TensorIteratorBase\n  ufunc_inner_loop:\n    Generic: add (AllAndComplex, BFloat16, Half, ComplexHalf)\n    ScalarOnly: add (Bool)\n  type_alias:\n    T0: [Float, Double]\n    T1: [Double, Int]\n  dim_order_alias:\n    D0: [0, 1, 2, 3]\n    D1: [0, 3, 2, 1]\n  kernels:\n    - arg_meta: null\n      kernel_name: default_impl\n    - arg_meta:\n        self: [T0, D0]\n        other: [T1, D0]\n        out: [T0, D0]\n      kernel_name: test_impl\n    - arg_meta:\n        self: [T1, D0]\n        other: [T1, D1]\n        out: [T0, D1]\n      kernel_name: test_impl_2\n\n- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor\n  device_check: NoCheck   # TensorIterator\n  structured_delegate: add.out\n  variants: function, method\n  tags: core\n\n- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)\n  device_check: NoCheck   # TensorIterator\n  structured: True\n  structured_inherits: TensorIteratorBase\n  type_alias:\n    T0: [Float]\n    T1: [Double]\n  dim_order_alias:\n    D0: [0, 1, 2, 3]\n  kernels:\n    - arg_meta: null\n      kernel_name: default_impl\n    - arg_meta:\n        self: [T0, D0]\n        other: [T1, D0]\n        out: [T0, D0]\n      kernel_name: test_impl\n\n- func: mul.Tensor(Tensor self, Tensor other) -> Tensor\n  device_check: NoCheck   # TensorIterator\n  structured_delegate: mul.out\n  variants: function, method\n  tags: core\n\n'

class TestParseNativeYaml(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.mkdtemp()
        self.aten_yaml_path = os.path.join(self.temp_dir, 'test_native_functions.yaml')
        with open(self.aten_yaml_path, 'w') as f:
            f.write(TEST_YAML)
        self.ops_yaml_path = os.path.join(self.temp_dir, 'test.yaml')
        self.tags_yaml_path = os.path.join(self.temp_dir, 'tags.yaml')
        with open(self.tags_yaml_path, 'w') as f:
            f.write('\n- tag: core\n  desc: test\n            ')
        with open(self.ops_yaml_path, 'w') as f:
            f.write('\n- op: add.out\n  device_check: NoCheck   # TensorIterator\n  dispatch:\n    CPU: torch::executor::add_out_kernel\n\n- op: mul.out\n  device_check: NoCheck   # TensorIterator\n  dispatch:\n    CPU: torch::executor::mul_out_kernel\n                ')

    def test_translate_native_yaml_writes_correct_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        out_yaml_path = os.path.join(self.temp_dir, 'out.yaml')
        with open(out_yaml_path, 'w') as out_file:
            translate_native_yaml(tags_yaml_path=self.tags_yaml_path, aten_yaml_path=self.aten_yaml_path, native_yaml_path=self.ops_yaml_path, use_aten_lib=False, out_file=out_file)
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        self.assertTrue(all(('func' in e for e in es)))
        self.assertTrue(all((e.get('variants') == 'function' for e in es)))
        for e in es:
            self.assertFalse({'kernels', 'type_alias', 'dim_order_alias'} < e.keys())

    def test_parse_yaml_files(self) -> None:
        if False:
            i = 10
            return i + 15
        custom_ops_yaml_path = None
        selector = SelectiveBuilder.get_nop_selector()
        use_aten_lib = False
        (parsed_yaml, custom_ops_parsed_yaml) = parse_yaml_files(aten_yaml_path=self.aten_yaml_path, tags_yaml_path=self.tags_yaml_path, native_yaml_path=self.ops_yaml_path, custom_ops_yaml_path=custom_ops_yaml_path, selector=selector, use_aten_lib=use_aten_lib)
        expected_kernel_entry = {'add.out': 1, 'mul.out': 1}
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))
        op_entries = parsed_yaml.kernel_index.index
        for (op_name, kernel_mapping) in op_entries.items():
            self.assertTrue(len(kernel_mapping) == expected_kernel_entry.pop(str(op_name)))
        self.assertTrue(len(expected_kernel_entry) == 0)

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass

class TestParseKernelYamlFiles(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.mkdtemp()
        self.aten_kernel_yaml_path = os.path.join(self.temp_dir, 'test_kernel_native_functions.yaml')
        with open(self.aten_kernel_yaml_path, 'w') as f:
            f.write(TEST_KERNEL_YAML)
        self.ops_yaml_path = os.path.join(self.temp_dir, 'test.yaml')
        self.tags_yaml_path = os.path.join(self.temp_dir, 'tags.yaml')
        with open(self.tags_yaml_path, 'w') as f:
            f.write('\n- tag: core\n  desc: test\n            ')
        with open(self.ops_yaml_path, 'w') as f:
            f.write('\n- op: add.out\n  device_check: NoCheck   # TensorIterator\n  dispatch:\n    CPU: torch::executor::add_out_kernel\n\n- op: mul.out\n  device_check: NoCheck   # TensorIterator\n  dispatch:\n    CPU: torch::executor::mul_out_kernel\n                ')

    def test_translate_kernel_native_yaml_writes_correct_data(self) -> None:
        if False:
            i = 10
            return i + 15
        out_yaml_path = os.path.join(self.temp_dir, 'out2.yaml')
        with open(out_yaml_path, 'w') as out_file:
            translate_native_yaml(tags_yaml_path=self.tags_yaml_path, aten_yaml_path=self.aten_kernel_yaml_path, native_yaml_path=self.ops_yaml_path, use_aten_lib=False, out_file=out_file)
        with open(out_yaml_path) as out_file:
            es = yaml.load(out_file, Loader=LineLoader)
        self.assertTrue(all(('func' in e for e in es)))
        self.assertTrue(all((e.get('variants') == 'function' for e in es)))
        for e in es:
            self.assertTrue({'kernels', 'type_alias', 'dim_order_alias'} < e.keys())

    def test_parse_yaml_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        custom_ops_yaml_path = None
        selector = SelectiveBuilder.get_nop_selector()
        use_aten_lib = False
        (parsed_yaml, custom_ops_parsed_yaml) = parse_yaml_files(aten_yaml_path=self.aten_kernel_yaml_path, tags_yaml_path=self.tags_yaml_path, native_yaml_path=self.ops_yaml_path, custom_ops_yaml_path=custom_ops_yaml_path, selector=selector, use_aten_lib=use_aten_lib)
        expected_kernel_entry = {'add.out': 9, 'mul.out': 2}
        self.assertTrue(len(parsed_yaml.native_functions) == len(expected_kernel_entry))
        op_entries = parsed_yaml.kernel_index.index
        for (op_name, kernel_mapping) in op_entries.items():
            self.assertTrue(len(kernel_mapping) == expected_kernel_entry.pop(str(op_name)))
        self.assertTrue(len(expected_kernel_entry) == 0)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass

class TestGenFunctionsDeclarations(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        (self.custom_1_native_function, custom_1_backend_index) = NativeFunction.from_yaml({'func': 'custom_1::op_1() -> bool', 'dispatch': {'CPU': 'kernel_1'}}, loc=Location(__file__, 1), valid_tags=set())
        (self.custom_2_native_function, custom_2_backend_index) = NativeFunction.from_yaml({'func': 'custom_2::op_2() -> bool', 'dispatch': {'CPU': 'kernel_2'}}, loc=Location(__file__, 1), valid_tags=set())
        backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = {DispatchKey.CPU: {}, DispatchKey.QuantizedCPU: {}}
        BackendIndex.grow_index(backend_indices, custom_1_backend_index)
        BackendIndex.grow_index(backend_indices, custom_2_backend_index)
        self.static_dispatch_idx = [BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, device_guard=False, index=backend_indices[k]) for k in backend_indices]
        self.kernel_index = ETKernelIndex.from_backend_indices(backend_indices)

    def test_operators_with_different_namespaces_are_grouped_correctly(self) -> None:
        if False:
            print('Hello World!')
        declarations = gen_functions_declarations(native_functions=[self.custom_1_native_function, self.custom_2_native_function], kernel_index=self.kernel_index, selector=SelectiveBuilder.get_nop_selector(), use_aten_lib=False)
        self.assertTrue('\nnamespace custom_1 {\n\n// custom_1::op_1() -> bool\nTORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {\n    return ::at::native::kernel_1(context);\n}\n\n} // namespace custom_1\n' in declarations)
        self.assertTrue('\nnamespace custom_2 {\n\n// custom_2::op_2() -> bool\nTORCH_API inline bool op_2(torch::executor::KernelRuntimeContext & context) {\n    return ::at::native::kernel_2(context);\n}\n\n} // namespace custom_2\n        ' in declarations)

    def test_aten_lib_has_context_arg(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        declarations = gen_functions_declarations(native_functions=[self.custom_1_native_function], kernel_index=self.kernel_index, selector=SelectiveBuilder.get_nop_selector(), use_aten_lib=True)
        self.assertTrue('\nnamespace custom_1 {\n\n// custom_1::op_1() -> bool\nTORCH_API inline bool op_1(torch::executor::KernelRuntimeContext & context) {\n    return at::op_1();\n}\n\n} // namespace custom_1\n        ' in declarations)

class TestComputeCodegenUnboxedKernels(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        (self.native_function_no_kern, _) = NativeFunction.from_yaml({'func': 'custom_1::op_1() -> bool', 'dispatch': {'CPU': 'unused_kernel_1'}}, loc=Location(__file__, 1), valid_tags=set())
        self.default_kernel_key = ETKernelKey(default=True)
        self.default_backend_metadata = BackendMetadata('default_kernel', False, 'at::native')
        self.default_kernel_entry = ([self.default_kernel_key], self.default_backend_metadata)

    def test_codegen_unboxed_specialized(self) -> None:
        if False:
            while True:
                i = 10
        specialized_kernel_key = ETKernelKey.gen_from_yaml({'self': ('T0', 'D0'), 'other': ('T0', 'D0'), 'out': ('T0', 'D0')}, {'T0': ['Double']}, {'D0': [0, 1, 2, 3]})
        selector = SelectiveBuilder.from_yaml_dict({'include_all_operators': True, 'et_kernel_metadata': {'custom_1::op_1': ['v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3']}})
        use_aten_lib = False
        entry = (self.native_function_no_kern, (specialized_kernel_key, self.default_backend_metadata))
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        expected_str = '\nKernel(\n    "custom_1::op_1",\n    "v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3",\n    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {\n        ' + '\n\n        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");\n        EXECUTORCH_SCOPE_PROF("native_call_op_1");\n        bool result_ = at::native::default_kernel(context, );\n\n        *stack[0] = EValue(result_);\n    }\n),\n'
        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_specialized_not_matching(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        specialized_kernel_key = ETKernelKey.gen_from_yaml({'self': ('T0', 'D0'), 'other': ('T0', 'D0'), 'out': ('T0', 'D0')}, {'T0': ['Double']}, {'D0': [0, 1, 2, 3]})
        selector = SelectiveBuilder.from_yaml_dict({'include_all_operators': True, 'et_kernel_metadata': {'custom_1::op_1': ['v1/8;0,1,2,3|7;0,1,2,3|7;0,1,2,3']}})
        use_aten_lib = False
        entry = (self.native_function_no_kern, (specialized_kernel_key, self.default_backend_metadata))
        self.assertRaises(Exception, ComputeCodegenUnboxedKernels(selector, use_aten_lib), entry)

    def test_codegen_unboxed_specialized_missing_root_op(self) -> None:
        if False:
            print('Hello World!')
        specialized_kernel_key = ETKernelKey.gen_from_yaml({'self': ('T0', 'D0'), 'other': ('T0', 'D0'), 'out': ('T0', 'D0')}, {'T0': ['Double']}, {'D0': [0, 1, 2, 3]})
        selector = SelectiveBuilder.from_yaml_dict({'et_kernel_metadata': {'custom_1::op_1': ['v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3']}})
        use_aten_lib = False
        entry = (self.native_function_no_kern, (specialized_kernel_key, self.default_backend_metadata))
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        expected_str = ''
        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_default(self) -> None:
        if False:
            return 10
        '\n        This test checks that if there is no specialized kernel, the default kernel is used.\n        '
        selector = SelectiveBuilder.from_yaml_dict({'include_all_operators': True, 'et_kernel_metadata': {'custom_1::op_1': ['v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3']}})
        use_aten_lib = False
        entry = (self.native_function_no_kern, self.default_kernel_entry)
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        expected_str = '\nKernel(\n    "custom_1::op_1",\n    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {\n        ' + '\n\n        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");\n        EXECUTORCH_SCOPE_PROF("native_call_op_1");\n        bool result_ = at::native::default_kernel(context, );\n\n        *stack[0] = EValue(result_);\n    }\n),\n'
        self.assertEqual(expected_str, result)

    def test_codegen_unboxed_default_kernel_key_selected(self) -> None:
        if False:
            while True:
                i = 10
        '\n        This test checks that if there is no specialized kernel, the default kernel is used, when the selector only has default key.\n        '
        selector = SelectiveBuilder.from_yaml_dict({'include_all_operators': True, 'et_kernel_metadata': {'custom_1::op_1': ['default']}})
        use_aten_lib = False
        entry = (self.native_function_no_kern, self.default_kernel_entry)
        result = ComputeCodegenUnboxedKernels(selector, use_aten_lib)(entry)
        expected_str = '\nKernel(\n    "custom_1::op_1",\n    [](torch::executor::KernelRuntimeContext & context, EValue** stack) {\n        ' + '\n\n        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_op_1");\n        EXECUTORCH_SCOPE_PROF("native_call_op_1");\n        bool result_ = at::native::default_kernel(context, );\n\n        *stack[0] = EValue(result_);\n    }\n),\n'
        self.assertEqual(expected_str, result)