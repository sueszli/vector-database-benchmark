import sys
import os
import argparse
import importlib.util

def get_processor():
    if False:
        while True:
            i = 10
    conv_template_path = os.path.join(os.path.dirname(__file__), '..', 'distutils', 'conv_template.py')
    spec = importlib.util.spec_from_file_location('conv_template', conv_template_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.process_file

def process_and_write_file(fromfile, outfile):
    if False:
        i = 10
        return i + 15
    'Process tempita templated file and write out the result.\n\n    The template file is expected to end in `.src`\n    (e.g., `.c.src` or `.h.src`).\n    Processing `npy_somefile.c.src` generates `npy_somefile.c`.\n\n    '
    process_file = get_processor()
    content = process_file(fromfile)
    with open(outfile, 'w') as f:
        f.write(content)

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the input file')
    parser.add_argument('-o', '--outfile', type=str, help='Path to the output file')
    parser.add_argument('-i', '--ignore', type=str, help='An ignored input - may be useful to add a dependency between custom targets')
    args = parser.parse_args()
    if not args.infile.endswith('.src'):
        raise ValueError(f'Unexpected extension: {args.infile}')
    outfile_abs = os.path.join(os.getcwd(), args.outfile)
    process_and_write_file(args.infile, outfile_abs)
if __name__ == '__main__':
    main()