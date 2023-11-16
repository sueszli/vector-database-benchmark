"""This is a Python API fuzzer for tf.raw_ops.Add."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf

def TestOneInput(data):
    if False:
        i = 10
        return i + 15
    'Test numeric randomized fuzzing input for tf.raw_ops.Add.'
    fh = FuzzingHelper(data)
    input_tensor_x = fh.get_random_numeric_tensor()
    input_tensor_y = fh.get_random_numeric_tensor()
    try:
        _ = tf.raw_ops.Add(x=input_tensor_x, y=input_tensor_y)
    except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError):
        pass

def main():
    if False:
        while True:
            i = 10
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()