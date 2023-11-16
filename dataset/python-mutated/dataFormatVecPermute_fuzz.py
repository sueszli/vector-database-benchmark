"""This is a Python API fuzzer for tf.raw_ops.DataFormatVecPermute."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf

@atheris.instrument_func
def TestOneInput(input_bytes):
    if False:
        return 10
    'Test randomized integer fuzzing input for tf.raw_ops.DataFormatVecPermute.'
    fh = FuzzingHelper(input_bytes)
    dtype = fh.get_tf_dtype()
    shape = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
    seed = fh.get_int()
    try:
        x = tf.random.uniform(shape=shape, dtype=dtype, seed=seed)
        src_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
        dest_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
        _ = tf.raw_ops.DataFormatVecPermute(x, src_format=src_format_digits, dst_format=dest_format_digits, name=fh.get_string())
    except (tf.errors.InvalidArgumentError, ValueError, TypeError):
        pass

def main():
    if False:
        for i in range(10):
            print('nop')
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()