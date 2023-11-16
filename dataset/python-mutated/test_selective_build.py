import unittest
from torchgen.selective_build.operator import *
from torchgen.model import Location, NativeFunction
from torchgen.selective_build.selector import combine_selective_builders, SelectiveBuilder

class TestSelectiveBuild(unittest.TestCase):

    def test_selective_build_operator(self):
        if False:
            for i in range(10):
                print('nop')
        op = SelectiveBuildOperator('aten::add.int', is_root_operator=True, is_used_for_training=False, include_all_overloads=False, _debug_info=None)
        self.assertTrue(op.is_root_operator)
        self.assertFalse(op.is_used_for_training)
        self.assertFalse(op.include_all_overloads)

    def test_selector_factory(self):
        if False:
            return 10
        yaml_config_v1 = '\ndebug_info:\n  - model1@v100\n  - model2@v51\noperators:\n  aten::add:\n    is_used_for_training: No\n    is_root_operator: Yes\n    include_all_overloads: Yes\n  aten::add.int:\n    is_used_for_training: Yes\n    is_root_operator: No\n    include_all_overloads: No\n  aten::mul.int:\n    is_used_for_training: Yes\n    is_root_operator: No\n    include_all_overloads: No\n'
        yaml_config_v2 = '\ndebug_info:\n  - model1@v100\n  - model2@v51\noperators:\n  aten::sub:\n    is_used_for_training: No\n    is_root_operator: Yes\n    include_all_overloads: No\n    debug_info:\n      - model1@v100\n  aten::sub.int:\n    is_used_for_training: Yes\n    is_root_operator: No\n    include_all_overloads: No\n'
        yaml_config_all = 'include_all_operators: Yes'
        yaml_config_invalid = 'invalid:'
        selector1 = SelectiveBuilder.from_yaml_str(yaml_config_v1)
        self.assertTrue(selector1.is_operator_selected('aten::add'))
        self.assertTrue(selector1.is_operator_selected('aten::add.int'))
        self.assertTrue(selector1.is_operator_selected('aten::add.float'))

        def gen():
            if False:
                print('Hello World!')
            return SelectiveBuilder.from_yaml_str(yaml_config_invalid)
        self.assertRaises(Exception, gen)
        selector_all = SelectiveBuilder.from_yaml_str(yaml_config_all)
        self.assertTrue(selector_all.is_operator_selected('aten::add'))
        self.assertTrue(selector_all.is_operator_selected('aten::sub'))
        self.assertTrue(selector_all.is_operator_selected('aten::sub.int'))
        self.assertTrue(selector_all.is_kernel_dtype_selected('add_kernel', 'int32'))
        selector2 = SelectiveBuilder.from_yaml_str(yaml_config_v2)
        self.assertFalse(selector2.is_operator_selected('aten::add'))
        self.assertTrue(selector2.is_operator_selected('aten::sub'))
        self.assertTrue(selector2.is_operator_selected('aten::sub.int'))
        selector_legacy_v1 = SelectiveBuilder.from_legacy_op_registration_allow_list(['aten::add', 'aten::add.int', 'aten::mul.int'], False, False)
        self.assertTrue(selector_legacy_v1.is_operator_selected('aten::add.float'))
        self.assertTrue(selector_legacy_v1.is_operator_selected('aten::add'))
        self.assertTrue(selector_legacy_v1.is_operator_selected('aten::add.int'))
        self.assertFalse(selector_legacy_v1.is_operator_selected('aten::sub'))
        self.assertFalse(selector_legacy_v1.is_root_operator('aten::add'))
        self.assertFalse(selector_legacy_v1.is_operator_selected_for_training('aten::add'))
        selector_legacy_v1 = SelectiveBuilder.from_legacy_op_registration_allow_list(['aten::add', 'aten::add.int', 'aten::mul.int'], True, False)
        self.assertTrue(selector_legacy_v1.is_root_operator('aten::add'))
        self.assertFalse(selector_legacy_v1.is_operator_selected_for_training('aten::add'))
        self.assertTrue(selector_legacy_v1.is_root_operator('aten::add.float'))
        self.assertFalse(selector_legacy_v1.is_operator_selected_for_training('aten::add.float'))
        selector_legacy_v1 = SelectiveBuilder.from_legacy_op_registration_allow_list(['aten::add', 'aten::add.int', 'aten::mul.int'], False, True)
        self.assertFalse(selector_legacy_v1.is_root_operator('aten::add'))
        self.assertTrue(selector_legacy_v1.is_operator_selected_for_training('aten::add'))
        self.assertFalse(selector_legacy_v1.is_root_operator('aten::add.float'))
        self.assertTrue(selector_legacy_v1.is_operator_selected_for_training('aten::add.float'))

    def test_operator_combine(self):
        if False:
            print('Hello World!')
        op1 = SelectiveBuildOperator('aten::add.int', is_root_operator=True, is_used_for_training=False, include_all_overloads=False, _debug_info=None)
        op2 = SelectiveBuildOperator('aten::add.int', is_root_operator=False, is_used_for_training=False, include_all_overloads=False, _debug_info=None)
        op3 = SelectiveBuildOperator('aten::add', is_root_operator=True, is_used_for_training=False, include_all_overloads=False, _debug_info=None)
        op4 = SelectiveBuildOperator('aten::add.int', is_root_operator=True, is_used_for_training=True, include_all_overloads=False, _debug_info=None)
        op5 = combine_operators(op1, op2)
        self.assertTrue(op5.is_root_operator)
        self.assertFalse(op5.is_used_for_training)
        op6 = combine_operators(op1, op4)
        self.assertTrue(op6.is_root_operator)
        self.assertTrue(op6.is_used_for_training)

        def gen_new_op():
            if False:
                for i in range(10):
                    print('nop')
            return combine_operators(op1, op3)
        self.assertRaises(Exception, gen_new_op)

    def test_training_op_fetch(self):
        if False:
            for i in range(10):
                print('nop')
        yaml_config = '\noperators:\n  aten::add.int:\n    is_used_for_training: No\n    is_root_operator: Yes\n    include_all_overloads: No\n  aten::add:\n    is_used_for_training: Yes\n    is_root_operator: No\n    include_all_overloads: Yes\n'
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        self.assertTrue(selector.is_operator_selected_for_training('aten::add.int'))
        self.assertTrue(selector.is_operator_selected_for_training('aten::add'))

    def test_kernel_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        yaml_config = '\nkernel_metadata:\n  add_kernel:\n    - int8\n    - int32\n  sub_kernel:\n    - int16\n    - int32\n  add/sub_kernel:\n    - float\n    - complex\n'
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int32'))
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int8'))
        self.assertFalse(selector.is_kernel_dtype_selected('add_kernel', 'int16'))
        self.assertFalse(selector.is_kernel_dtype_selected('add1_kernel', 'int32'))
        self.assertFalse(selector.is_kernel_dtype_selected('add_kernel', 'float'))
        self.assertTrue(selector.is_kernel_dtype_selected('add/sub_kernel', 'float'))
        self.assertTrue(selector.is_kernel_dtype_selected('add/sub_kernel', 'complex'))
        self.assertFalse(selector.is_kernel_dtype_selected('add/sub_kernel', 'int16'))
        self.assertFalse(selector.is_kernel_dtype_selected('add/sub_kernel', 'int32'))

    def test_merge_kernel_dtypes(self):
        if False:
            i = 10
            return i + 15
        yaml_config1 = '\nkernel_metadata:\n  add_kernel:\n    - int8\n  add/sub_kernel:\n    - float\n    - complex\n    - none\n  mul_kernel:\n    - int8\n'
        yaml_config2 = '\nkernel_metadata:\n  add_kernel:\n    - int32\n  sub_kernel:\n    - int16\n    - int32\n  add/sub_kernel:\n    - float\n    - complex\n'
        selector1 = SelectiveBuilder.from_yaml_str(yaml_config1)
        selector2 = SelectiveBuilder.from_yaml_str(yaml_config2)
        selector = combine_selective_builders(selector1, selector2)
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int32'))
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int8'))
        self.assertFalse(selector.is_kernel_dtype_selected('add_kernel', 'int16'))
        self.assertFalse(selector.is_kernel_dtype_selected('add1_kernel', 'int32'))
        self.assertFalse(selector.is_kernel_dtype_selected('add_kernel', 'float'))
        self.assertTrue(selector.is_kernel_dtype_selected('add/sub_kernel', 'float'))
        self.assertTrue(selector.is_kernel_dtype_selected('add/sub_kernel', 'complex'))
        self.assertTrue(selector.is_kernel_dtype_selected('add/sub_kernel', 'none'))
        self.assertFalse(selector.is_kernel_dtype_selected('add/sub_kernel', 'int16'))
        self.assertFalse(selector.is_kernel_dtype_selected('add/sub_kernel', 'int32'))
        self.assertTrue(selector.is_kernel_dtype_selected('mul_kernel', 'int8'))
        self.assertFalse(selector.is_kernel_dtype_selected('mul_kernel', 'int32'))

    def test_all_kernel_dtypes_selected(self):
        if False:
            for i in range(10):
                print('nop')
        yaml_config = '\ninclude_all_non_op_selectives: True\n'
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int32'))
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int8'))
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'int16'))
        self.assertTrue(selector.is_kernel_dtype_selected('add1_kernel', 'int32'))
        self.assertTrue(selector.is_kernel_dtype_selected('add_kernel', 'float'))

    def test_custom_namespace_selected_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        yaml_config = '\noperators:\n  aten::add.int:\n    is_used_for_training: No\n    is_root_operator: Yes\n    include_all_overloads: No\n  custom::add:\n    is_used_for_training: Yes\n    is_root_operator: No\n    include_all_overloads: Yes\n'
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        (native_function, _) = NativeFunction.from_yaml({'func': 'custom::add() -> Tensor'}, loc=Location(__file__, 1), valid_tags=set())
        self.assertTrue(selector.is_native_function_selected(native_function))

class TestExecuTorchSelectiveBuild(unittest.TestCase):

    def test_et_kernel_selected(self):
        if False:
            i = 10
            return i + 15
        yaml_config = '\net_kernel_metadata:\n  aten::add.out:\n   - "v1/6;0,1|6;0,1|6;0,1|6;0,1"\n  aten::sub.out:\n   - "v1/6;0,1|6;0,1|6;0,1|6;0,1"\n'
        selector = SelectiveBuilder.from_yaml_str(yaml_config)
        self.assertListEqual(['v1/6;0,1|6;0,1|6;0,1|6;0,1'], selector.et_get_selected_kernels('aten::add.out', ['v1/6;0,1|6;0,1|6;0,1|6;0,1', 'v1/3;0,1|3;0,1|3;0,1|3;0,1', 'v1/6;1,0|6;0,1|6;0,1|6;0,1']))
        self.assertListEqual(['v1/6;0,1|6;0,1|6;0,1|6;0,1'], selector.et_get_selected_kernels('aten::sub.out', ['v1/6;0,1|6;0,1|6;0,1|6;0,1']))
        self.assertListEqual([], selector.et_get_selected_kernels('aten::mul.out', ['v1/6;0,1|6;0,1|6;0,1|6;0,1']))
        self.assertListEqual(['v2/6;0,1|6;0,1|6;0,1|6;0,1'], selector.et_get_selected_kernels('aten::add.out', ['v2/6;0,1|6;0,1|6;0,1|6;0,1']))