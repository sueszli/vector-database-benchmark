"""This is a Python API fuzzer for tf.raw_ops.Acosh."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf

def TestOneInput(data):
    if False:
        i = 10
        return i + 15
    'Test randomized fuzzing input for tf.raw_ops.Acosh.'
    fh = FuzzingHelper(data)
    dtype = fh.get_tf_dtype(allowed_set=[tf.float16, tf.float32, tf.float64])
    input_tensor = fh.get_random_numeric_tensor(dtype=dtype)
    _ = tf.raw_ops.Acosh(x=input_tensor)

def main():
    if False:
        i = 10
        return i + 15
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()