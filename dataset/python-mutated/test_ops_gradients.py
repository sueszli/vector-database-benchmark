from functools import partial
import torch
from torch.testing._internal.common_utils import TestGradients, run_tests, TestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.control_flow_opinfo_db import control_flow_opinfo_db
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops, OpDTypes
_gradcheck_ops = partial(ops, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble])

class TestBwdGradients(TestGradients):

    @_gradcheck_ops(op_db + control_flow_opinfo_db + custom_op_db)
    def test_fn_grad(self, device, dtype, op):
        if False:
            print('Hello World!')
        if dtype not in op.supported_backward_dtypes(torch.device(device).type):
            self.skipTest('Skipped! Dtype is not in supported backward dtypes!')
        else:
            self._grad_test_helper(device, dtype, op, op.get_op())

    @_gradcheck_ops(op_db + custom_op_db)
    def test_inplace_grad(self, device, dtype, op):
        if False:
            return 10
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant:
            self.skipTest('Op has no inplace variant!')
        if not op.supports_inplace_autograd:
            inplace = self._get_safe_inplace(op.get_inplace())
            for sample in op.sample_inputs(device, dtype, requires_grad=True):
                if sample.broadcasts_input:
                    continue
                with self.assertRaises(Exception):
                    result = inplace(sample)
                    result.sum().backward()
        else:
            self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    @_gradcheck_ops(op_db + control_flow_opinfo_db + custom_op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest("Op claims it doesn't support gradgrad. This is not verified.")
        else:
            self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

    @_gradcheck_ops(op_db + custom_op_db)
    def test_fn_fail_gradgrad(self, device, dtype, op):
        if False:
            while True:
                i = 10
        self._skip_helper(op, device, dtype)
        if op.supports_gradgrad:
            self.skipTest('Skipped! Operation does support gradgrad')
        err_msg = 'derivative for .* is not implemented'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

    @_gradcheck_ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest('Skipped! Operation does not support inplace autograd.')
        self._check_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), 'bwgrad_bwgrad')
instantiate_device_type_tests(TestBwdGradients, globals())
if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()