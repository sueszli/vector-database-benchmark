"""A template to define composite ops."""
import os
import sys
from absl import app
from tensorflow.compiler.mlir.tfr.python.composite import Composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('output', None, 'Path to write the genereated register op file and MLIR file.')
flags.DEFINE_bool('gen_register_op', True, 'Generate register op cc file or tfr mlir file.')
flags.mark_flag_as_required('output')

@Composite('TestRandom', derived_attrs=['T: numbertype'], outputs=['o: T'])
def _composite_random_op():
    if False:
        i = 10
        return i + 15
    pass

def main(_):
    if False:
        print('Hello World!')
    if FLAGS.gen_register_op:
        assert FLAGS.output.endswith('.cc')
        generated_code = gen_register_op(sys.modules[__name__], '_composite_')
    else:
        assert FLAGS.output.endswith('.mlir')
        generated_code = tfr_gen_from_module(sys.modules[__name__], '_composite_')
    dirname = os.path.dirname(FLAGS.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(FLAGS.output, 'w') as f:
        f.write(generated_code)
if __name__ == '__main__':
    app.run(main=main)