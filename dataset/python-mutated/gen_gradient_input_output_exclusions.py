"""Script to generate inputs/outputs exclusion lists for GradientTape.

To use this script:

bazel run tensorflow/python/eager:gen_gradient_input_output_exclusions -- \\
  $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc
"""
import argparse
from tensorflow.python.eager import gradient_input_output_exclusions

def main(output_file):
    if False:
        for i in range(10):
            print('nop')
    with open(output_file, 'w') as fp:
        fp.write(gradient_input_output_exclusions.get_contents())
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('output', metavar='O', type=str, help='Output file.')
    args = arg_parser.parse_args()
    main(args.output)