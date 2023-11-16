import fnmatch
import io
import shutil
import tempfile
import torch
import torch.utils.show_pickle
from torch.jit.mobile import _load_for_lite_interpreter, _get_mobile_model_contained_types, _get_model_bytecode_version, _get_model_ops_and_info, _backport_for_mobile_to_buffer, _backport_for_mobile
from torch.testing._internal.common_utils import TestCase, run_tests
from pathlib import Path
pytorch_test_dir = Path(__file__).resolve().parents[1]
SCRIPT_MODULE_V4_BYTECODE_PKL = "\n(4,\n ('__torch__.*.TestModule.forward',\n  (('instructions',\n    (('STOREN', 1, 2),\n     ('DROPR', 1, 0),\n     ('LOADC', 0, 0),\n     ('LOADC', 1, 0),\n     ('MOVE', 2, 0),\n     ('OP', 0, 0),\n     ('LOADC', 1, 0),\n     ('OP', 1, 0),\n     ('RET', 0, 0))),\n   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),\n   ('constants',\n    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, '0', 'cpu', 8),),\n       0,\n       (2, 4),\n       (4, 1),\n       False,\n       collections.OrderedDict()),\n     1)),\n   ('types', ()),\n   ('register_size', 2)),\n  (('arguments',\n    ((('name', 'self'),\n      ('type', '__torch__.*.TestModule'),\n      ('default_value', None)),\n     (('name', 'y'), ('type', 'int'), ('default_value', None)))),\n   ('returns',\n    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))\n        "
SCRIPT_MODULE_V5_BYTECODE_PKL = "\n(5,\n ('__torch__.*.TestModule.forward',\n  (('instructions',\n    (('STOREN', 1, 2),\n     ('DROPR', 1, 0),\n     ('LOADC', 0, 0),\n     ('LOADC', 1, 0),\n     ('MOVE', 2, 0),\n     ('OP', 0, 0),\n     ('LOADC', 1, 0),\n     ('OP', 1, 0),\n     ('RET', 0, 0))),\n   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),\n   ('constants',\n    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, 'constants/0', 'cpu', 8),),\n       0,\n       (2, 4),\n       (4, 1),\n       False,\n       collections.OrderedDict()),\n     1)),\n   ('types', ()),\n   ('register_size', 2)),\n  (('arguments',\n    ((('name', 'self'),\n      ('type', '__torch__.*.TestModule'),\n      ('default_value', None)),\n     (('name', 'y'), ('type', 'int'), ('default_value', None)))),\n   ('returns',\n    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))\n        "
SCRIPT_MODULE_V6_BYTECODE_PKL = "\n(6,\n ('__torch__.*.TestModule.forward',\n  (('instructions',\n    (('STOREN', 1, 2),\n     ('DROPR', 1, 0),\n     ('LOADC', 0, 0),\n     ('LOADC', 1, 0),\n     ('MOVE', 2, 0),\n     ('OP', 0, 0),\n     ('OP', 1, 0),\n     ('RET', 0, 0))),\n   ('operators', (('aten::add', 'int', 2), ('aten::add', 'Scalar', 2))),\n   ('constants',\n    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, '0', 'cpu', 8),),\n       0,\n       (2, 4),\n       (4, 1),\n       False,\n       collections.OrderedDict()),\n     1)),\n   ('types', ()),\n   ('register_size', 2)),\n  (('arguments',\n    ((('name', 'self'),\n      ('type', '__torch__.*.TestModule'),\n      ('default_value', None)),\n     (('name', 'y'), ('type', 'int'), ('default_value', None)))),\n   ('returns',\n    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))\n    "
SCRIPT_MODULE_BYTECODE_PKL = {4: {'bytecode_pkl': SCRIPT_MODULE_V4_BYTECODE_PKL, 'model_name': 'script_module_v4.ptl'}}
MINIMUM_TO_VERSION = 4

class testVariousModelVersions(TestCase):

    def test_get_model_bytecode_version(self):
        if False:
            i = 10
            return i + 15

        def check_model_version(model_path, expect_version):
            if False:
                i = 10
                return i + 15
            actual_version = _get_model_bytecode_version(model_path)
            assert actual_version == expect_version
        for (version, model_info) in SCRIPT_MODULE_BYTECODE_PKL.items():
            model_path = pytorch_test_dir / 'cpp' / 'jit' / model_info['model_name']
            check_model_version(model_path, version)

    def test_bytecode_values_for_all_backport_functions(self):
        if False:
            while True:
                i = 10
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        current_from_version = maximum_checked_in_model_version
        with tempfile.TemporaryDirectory() as tmpdirname:
            while current_from_version > MINIMUM_TO_VERSION:
                model_name = SCRIPT_MODULE_BYTECODE_PKL[current_from_version]['model_name']
                input_model_path = pytorch_test_dir / 'cpp' / 'jit' / model_name
                tmp_output_model_path_backport = Path(tmpdirname, 'tmp_script_module_backport.ptl')
                current_to_version = current_from_version - 1
                backport_success = _backport_for_mobile(input_model_path, tmp_output_model_path_backport, current_to_version)
                assert backport_success
                expect_bytecode_pkl = SCRIPT_MODULE_BYTECODE_PKL[current_to_version]['bytecode_pkl']
                buf = io.StringIO()
                torch.utils.show_pickle.main(['', tmpdirname + '/' + tmp_output_model_path_backport.name + '@*/bytecode.pkl'], output_stream=buf)
                output = buf.getvalue()
                acutal_result_clean = ''.join(output.split())
                expect_result_clean = ''.join(expect_bytecode_pkl.split())
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                assert isMatch
                current_from_version -= 1
            shutil.rmtree(tmpdirname)

    def test_backport_bytecode_from_file_to_file(self):
        if False:
            return 10
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        script_module_v5_path = pytorch_test_dir / 'cpp' / 'jit' / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]['model_name']
        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_backport_model_path = Path(tmpdirname, 'tmp_script_module_v5_backported_to_v4.ptl')
                success = _backport_for_mobile(script_module_v5_path, tmp_backport_model_path, maximum_checked_in_model_version - 1)
                assert success
                buf = io.StringIO()
                torch.utils.show_pickle.main(['', tmpdirname + '/' + tmp_backport_model_path.name + '@*/bytecode.pkl'], output_stream=buf)
                output = buf.getvalue()
                expected_result = SCRIPT_MODULE_V4_BYTECODE_PKL
                acutal_result_clean = ''.join(output.split())
                expect_result_clean = ''.join(expected_result.split())
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                assert isMatch
                mobile_module = _load_for_lite_interpreter(str(tmp_backport_model_path))
                module_input = 1
                mobile_module_result = mobile_module(module_input)
                expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
                torch.testing.assert_close(mobile_module_result, expected_mobile_module_result)
                shutil.rmtree(tmpdirname)

    def test_backport_bytecode_from_file_to_buffer(self):
        if False:
            return 10
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        script_module_v5_path = pytorch_test_dir / 'cpp' / 'jit' / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]['model_name']
        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            script_module_v4_buffer = _backport_for_mobile_to_buffer(script_module_v5_path, maximum_checked_in_model_version - 1)
            buf = io.StringIO()
            bytesio = io.BytesIO(script_module_v4_buffer)
            backport_version = _get_model_bytecode_version(bytesio)
            assert backport_version == maximum_checked_in_model_version - 1
            bytesio = io.BytesIO(script_module_v4_buffer)
            mobile_module = _load_for_lite_interpreter(bytesio)
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_close(mobile_module_result, expected_mobile_module_result)

    def test_get_model_ops_and_info(self):
        if False:
            i = 10
            return i + 15
        script_module_v6 = pytorch_test_dir / 'cpp' / 'jit' / 'script_module_v6.ptl'
        ops_v6 = _get_model_ops_and_info(script_module_v6)
        assert ops_v6['aten::add.int'].num_schema_args == 2
        assert ops_v6['aten::add.Scalar'].num_schema_args == 2

    def test_get_mobile_model_contained_types(self):
        if False:
            return 10

        class MyTestModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + 10
        sample_input = torch.tensor([1])
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input)
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        type_list = _get_mobile_model_contained_types(buffer)
        assert len(type_list) >= 0
if __name__ == '__main__':
    run_tests()