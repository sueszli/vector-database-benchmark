"""A test file for run_python_test.py."""
import sys
from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('print_value', 'hello world', 'String to print.')
flags.DEFINE_integer('return_value', 0, 'Return value for the process.')

def main(argv):
    if False:
        while True:
            i = 10
    print('Num args:', len(argv))
    print('argv[0]:', argv[0])
    print('print_value:', FLAGS.print_value)
    print('return_value:', FLAGS.return_value)
    sys.exit(FLAGS.return_value)
if __name__ == '__main__':
    app.run(main)