"""This is a Python API fuzzer for tf.raw_ops.Acos."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf

def TestOneInput(data):
    if False:
        print('Hello World!')
    'Test randomized fuzzing input for tf.raw_ops.Acos.'
    fh = FuzzingHelper(data)
    input_tensor = fh.get_random_numeric_tensor()
    _ = tf.raw_ops.Acos(x=input_tensor)

def main():
    if False:
        return 10
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()