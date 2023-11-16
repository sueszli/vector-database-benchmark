"""Demonstrates how the composition overrides the behavior of an existing op."""
import os
import sys
from absl import app
from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_array_ops as array_ops
from tensorflow.python.platform import flags
Composite = composite.Composite
FLAGS = flags.FLAGS
flags.DEFINE_string('output', None, 'Path to write the genereated register op file and MLIR file.')
flags.DEFINE_bool('gen_register_op', True, 'Generate register op cc file or tfr mlir file.')

@Composite('TestAttr')
def _override_test_attr_op():
    if False:
        print('Hello World!')
    ret = array_ops.Const(value=100.0, dtype=dtypes.float32)
    return ret

def main(_):
    if False:
        for i in range(10):
            print('nop')
    if FLAGS.gen_register_op:
        assert FLAGS.output.endswith('.cc')
        generated_code = gen_register_op(sys.modules[__name__], '_override_')
    else:
        assert FLAGS.output.endswith('.mlir')
        generated_code = tfr_gen_from_module(sys.modules[__name__], '_override_')
    dirname = os.path.dirname(FLAGS.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(FLAGS.output, 'w') as f:
        f.write(generated_code)
if __name__ == '__main__':
    app.run(main=main)