import os
import re
import yaml
import textwrap
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from collections import namedtuple
path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
all_operators_with_namedtuple_return = {'max', 'min', 'aminmax', 'median', 'nanmedian', 'mode', 'kthvalue', 'svd', 'qr', 'geqrf', 'slogdet', 'sort', 'topk', 'linalg_inv_ex', 'triangular_solve', 'cummax', 'cummin', 'linalg_eigh', '_linalg_eigh', '_unpack_dual', 'linalg_qr', 'linalg_svd', '_linalg_svd', 'linalg_slogdet', '_linalg_slogdet', 'fake_quantize_per_tensor_affine_cachemask', 'fake_quantize_per_channel_affine_cachemask', 'linalg_lstsq', 'linalg_eig', 'linalg_cholesky_ex', 'frexp', 'lu_unpack', 'histogram', 'histogramdd', '_fake_quantize_per_tensor_affine_cachemask_tensor_qparams', '_fused_moving_avg_obs_fq_helper', 'linalg_lu_factor', 'linalg_lu_factor_ex', 'linalg_lu', '_linalg_det', '_lu_with_info', 'linalg_ldl_factor_ex', 'linalg_ldl_factor', 'linalg_solve_ex', '_linalg_solve_ex'}
all_operators_with_namedtuple_return_skip_list = {'_scaled_dot_product_flash_attention', '_scaled_dot_product_efficient_attention'}

class TestNamedTupleAPI(TestCase):

    def test_import_return_types(self):
        if False:
            while True:
                i = 10
        import torch.return_types
        exec('from torch.return_types import *')

    def test_native_functions_yaml(self):
        if False:
            print('Hello World!')
        operators_found = set()
        regex = re.compile('^(\\w*)(\\(|\\.)')
        with open(aten_native_yaml) as file:
            for f in yaml.safe_load(file.read()):
                f = f['func']
                ret = f.split('->')[1].strip()
                name = regex.findall(f)[0][0]
                if name in all_operators_with_namedtuple_return:
                    operators_found.add(name)
                    continue
                if '_backward' in name or name.endswith('_forward'):
                    continue
                if not ret.startswith('('):
                    continue
                if ret == '()':
                    continue
                if name in all_operators_with_namedtuple_return_skip_list:
                    continue
                ret = ret[1:-1].split(',')
                for r in ret:
                    r = r.strip()
                    self.assertEqual(len(r.split()), 1, 'only allowlisted operators are allowed to have named return type, got ' + name)
        self.assertEqual(all_operators_with_namedtuple_return, operators_found, textwrap.dedent('\n        Some elements in the `all_operators_with_namedtuple_return` of test_namedtuple_return_api.py\n        could not be found. Do you forget to update test_namedtuple_return_api.py after renaming some\n        operator?\n        '))

    def test_namedtuple_return(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.randn(5, 5)
        per_channel_scale = torch.randn(5)
        per_channel_zp = torch.zeros(5, dtype=torch.int64)
        op = namedtuple('op', ['operators', 'input', 'names', 'hasout'])
        operators = [op(operators=['max', 'min', 'median', 'nanmedian', 'mode', 'sort', 'topk', 'cummax', 'cummin'], input=(0,), names=('values', 'indices'), hasout=True), op(operators=['kthvalue'], input=(1, 0), names=('values', 'indices'), hasout=True), op(operators=['svd'], input=(), names=('U', 'S', 'V'), hasout=True), op(operators=['linalg_svd', '_linalg_svd'], input=(), names=('U', 'S', 'Vh'), hasout=True), op(operators=['slogdet', 'linalg_slogdet'], input=(), names=('sign', 'logabsdet'), hasout=True), op(operators=['_linalg_slogdet'], input=(), names=('sign', 'logabsdet', 'LU', 'pivots'), hasout=True), op(operators=['qr', 'linalg_qr'], input=(), names=('Q', 'R'), hasout=True), op(operators=['geqrf'], input=(), names=('a', 'tau'), hasout=True), op(operators=['triangular_solve'], input=(a,), names=('solution', 'cloned_coefficient'), hasout=True), op(operators=['linalg_eig'], input=(), names=('eigenvalues', 'eigenvectors'), hasout=True), op(operators=['linalg_eigh'], input=('L',), names=('eigenvalues', 'eigenvectors'), hasout=True), op(operators=['_linalg_eigh'], input=('L',), names=('eigenvalues', 'eigenvectors'), hasout=True), op(operators=['linalg_cholesky_ex'], input=(), names=('L', 'info'), hasout=True), op(operators=['linalg_inv_ex'], input=(), names=('inverse', 'info'), hasout=True), op(operators=['linalg_solve_ex'], input=(a,), names=('result', 'info'), hasout=True), op(operators=['_linalg_solve_ex'], input=(a,), names=('result', 'LU', 'pivots', 'info'), hasout=True), op(operators=['linalg_lu_factor'], input=(), names=('LU', 'pivots'), hasout=True), op(operators=['linalg_lu_factor_ex'], input=(), names=('LU', 'pivots', 'info'), hasout=True), op(operators=['linalg_ldl_factor'], input=(), names=('LD', 'pivots'), hasout=True), op(operators=['linalg_ldl_factor_ex'], input=(), names=('LD', 'pivots', 'info'), hasout=True), op(operators=['linalg_lu'], input=(), names=('P', 'L', 'U'), hasout=True), op(operators=['fake_quantize_per_tensor_affine_cachemask'], input=(0.1, 0, 0, 255), names=('output', 'mask'), hasout=False), op(operators=['fake_quantize_per_channel_affine_cachemask'], input=(per_channel_scale, per_channel_zp, 1, 0, 255), names=('output', 'mask'), hasout=False), op(operators=['_unpack_dual'], input=(0,), names=('primal', 'tangent'), hasout=False), op(operators=['linalg_lstsq'], input=(a,), names=('solution', 'residuals', 'rank', 'singular_values'), hasout=False), op(operators=['frexp'], input=(), names=('mantissa', 'exponent'), hasout=True), op(operators=['lu_unpack'], input=(torch.tensor([3, 2, 1, 4, 5], dtype=torch.int32), True, True), names=('P', 'L', 'U'), hasout=True), op(operators=['histogram'], input=(1,), names=('hist', 'bin_edges'), hasout=True), op(operators=['histogramdd'], input=(1,), names=('hist', 'bin_edges'), hasout=False), op(operators=['_fake_quantize_per_tensor_affine_cachemask_tensor_qparams'], input=(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int), torch.tensor([1]), 0, 255), names=('output', 'mask'), hasout=False), op(operators=['_fused_moving_avg_obs_fq_helper'], input=(torch.tensor([1]), torch.tensor([1]), torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([1]), 0.01, 0, 255, 0), names=('output', 'mask'), hasout=False), op(operators=['_linalg_det'], input=(), names=('result', 'LU', 'pivots'), hasout=True), op(operators=['aminmax'], input=(), names=('min', 'max'), hasout=True), op(operators=['_lu_with_info'], input=(), names=('LU', 'pivots', 'info'), hasout=False)]

        def get_func(f):
            if False:
                while True:
                    i = 10
            "Return either torch.f or torch.linalg.f, where 'f' is a string"
            mod = torch
            if f.startswith('linalg_'):
                mod = torch.linalg
                f = f[7:]
            if f.startswith('_'):
                mod = torch._VF
            return getattr(mod, f, None)

        def check_namedtuple(tup, names):
            if False:
                print('Hello World!')
            "Check that the namedtuple 'tup' has the given names"
            for (i, name) in enumerate(names):
                self.assertIs(getattr(tup, name), tup[i])

        def check_torch_return_type(f, names):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Check that the return_type exists in torch.return_types\n            and they can constructed.\n            '
            return_type = getattr(torch.return_types, f)
            inputs = [torch.randn(()) for _ in names]
            self.assertEqual(type(return_type(inputs)), return_type)
        for op in operators:
            for f in op.operators:
                func = get_func(f)
                if func:
                    ret1 = func(a, *op.input)
                    check_namedtuple(ret1, op.names)
                    check_torch_return_type(f, op.names)
                if func and op.hasout:
                    ret2 = func(a, *op.input, out=tuple(ret1))
                    check_namedtuple(ret2, op.names)
                    check_torch_return_type(f + '_out', op.names)
                meth = getattr(a, f, None)
                if meth:
                    ret3 = meth(*op.input)
                    check_namedtuple(ret3, op.names)
        all_covered_operators = {x for y in operators for x in y.operators}
        self.assertEqual(all_operators_with_namedtuple_return, all_covered_operators, textwrap.dedent('\n        The set of covered operators does not match the `all_operators_with_namedtuple_return` of\n        test_namedtuple_return_api.py. Do you forget to add test for that operator?\n        '))
if __name__ == '__main__':
    run_tests()