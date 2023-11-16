"""Tests for `op_reg_gen` module."""
import sys
from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.python.platform import test
Composite = composite.Composite

@composite.Composite('TestNoOp', derived_attrs=['T: numbertype'], outputs=['o1: T'])
def _composite_no_op():
    if False:
        print('Hello World!')
    pass

@Composite('TestCompositeOp', inputs=['x: T', 'y: T'], attrs=['act: {"", "relu"}', 'trans: bool = true'], derived_attrs=['T: numbertype'], outputs=['o1: T', 'o2: T'])
def _composite_op(x, y, act, trans):
    if False:
        print('Hello World!')
    return (x + act, y + trans)

class TFRGenTensorTest(test.TestCase):
    """MLIR Generation Tests for MLIR TFR Program."""

    def test_op_reg_gen(self):
        if False:
            return 10
        cxx_code = gen_register_op(sys.modules[__name__])
        cxx_code_exp = '\n      CHECK: #include "tensorflow/core/framework/op.h"\n      CHECK-EMPTY\n      CHECK: namespace tensorflow {\n      CHECK-EMPTY\n      CHECK-LABEL: REGISTER_OP("TestNoOp")\n      CHECK-NEXT:      .Attr("T: numbertype")\n      CHECK-NEXT:      .Output("o1: T");\n      CHECK-EMPTY\n      CHECK-LABEL: REGISTER_OP("TestCompositeOp")\n      CHECK-NEXT:      .Input("x: T")\n      CHECK-NEXT:      .Input("y: T")\n      CHECK-NEXT:      .Attr("act: {\'\', \'relu\'}")\n      CHECK-NEXT:      .Attr("trans: bool = true")\n      CHECK-NEXT:      .Attr("T: numbertype")\n      CHECK-NEXT:      .Output("o1: T")\n      CHECK-NEXT:      .Output("o2: T");\n      CHECK-EMPTY\n      CHECK:  }  // namespace tensorflow\n    '
        self.assertTrue(fw.check(str(cxx_code), cxx_code_exp), str(cxx_code))
if __name__ == '__main__':
    test.main()