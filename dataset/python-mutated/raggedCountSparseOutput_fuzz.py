"""This is a Python API fuzzer for tf.raw_ops.RaggedCountSparseOutput."""
import atheris
with atheris.instrument_imports():
    import sys
    from python_fuzzing import FuzzingHelper
    import tensorflow as tf

@atheris.instrument_func
def TestOneInput(input_bytes):
    if False:
        print('Hello World!')
    'Test randomized integer/float fuzzing input for tf.raw_ops.RaggedCountSparseOutput.'
    fh = FuzzingHelper(input_bytes)
    splits = fh.get_int_list()
    values = fh.get_int_or_float_list()
    weights = fh.get_int_list()
    try:
        (_, _, _) = tf.raw_ops.RaggedCountSparseOutput(splits=splits, values=values, weights=weights, binary_output=False)
    except tf.errors.InvalidArgumentError:
        pass

def main():
    if False:
        return 10
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()