"""Generate a single synthetic sample."""
import io
import os
import numpy as np
import tensorflow as tf
import synthetic_model
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('sample_filename', None, 'Output file to store the generated binary code.')

def GenerateSample(filename, code_shape, layer_depth):
    if False:
        print('Hello World!')
    code = synthetic_model.GenerateSingleCode(code_shape)
    code = np.round(code)
    if code_shape[-1] % layer_depth != 0:
        raise ValueError('Number of layers is not an integer')
    height = code_shape[0]
    width = code_shape[1]
    code = code.reshape([1, height, width, -1, layer_depth])
    code = np.transpose(code, [3, 0, 1, 2, 4])
    int_codes = code.astype(np.int8)
    exported_codes = np.packbits(int_codes.reshape(-1))
    output = io.BytesIO()
    np.savez_compressed(output, shape=int_codes.shape, codes=exported_codes)
    with tf.gfile.FastGFile(filename, 'wb') as code_file:
        code_file.write(output.getvalue())

def main(argv=None):
    if False:
        for i in range(10):
            print('nop')
    layer_depth = 2
    GenerateSample(FLAGS.sample_filename, [31, 36, 8], layer_depth)
if __name__ == '__main__':
    tf.app.run()