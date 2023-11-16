"""This is a Python API fuzzer for tf.raw_ops.ImmutableConst."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf
_DEFAULT_FILENAME = '/tmp/test.txt'

@atheris.instrument_func
def TestOneInput(input_bytes):
    if False:
        i = 10
        return i + 15
    'Test randomized integer fuzzing input for tf.raw_ops.ImmutableConst.'
    fh = FuzzingHelper(input_bytes)
    dtype = fh.get_tf_dtype()
    shape = fh.get_int_list()
    try:
        with open(_DEFAULT_FILENAME, 'w') as f:
            f.write(fh.get_string())
        _ = tf.raw_ops.ImmutableConst(dtype=dtype, shape=shape, memory_region_name=_DEFAULT_FILENAME)
    except (tf.errors.InvalidArgumentError, tf.errors.InternalError, UnicodeEncodeError, UnicodeDecodeError):
        pass

def main():
    if False:
        print('Hello World!')
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()