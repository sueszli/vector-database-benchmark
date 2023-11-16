"""This is a Python API fuzzer for tf.constant."""
import atheris
with atheris.instrument_imports():
    import sys
    import tensorflow as tf

def TestOneInput(data):
    if False:
        i = 10
        return i + 15
    tf.constant(data)

def main():
    if False:
        for i in range(10):
            print('nop')
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
if __name__ == '__main__':
    main()