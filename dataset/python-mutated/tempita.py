import sys
import os
import argparse
from Cython import Tempita as tempita

def process_tempita(fromfile, outfile):
    if False:
        print('Hello World!')
    'Process tempita templated file and write out the result.\n\n    The template file is expected to end in `.c.in` or `.pyx.in`:\n    E.g. processing `template.c.in` generates `template.c`.\n\n    '
    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile, encoding=sys.getdefaultencoding())
    content = template.substitute()
    with open(outfile, 'w') as f:
        f.write(content)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the input file')
    parser.add_argument('-o', '--outdir', type=str, help='Path to the output directory')
    parser.add_argument('-i', '--ignore', type=str, help='An ignored input - may be useful to add a dependency between custom targets')
    args = parser.parse_args()
    if not args.infile.endswith('.in'):
        raise ValueError(f'Unexpected extension: {args.infile}')
    if os.path.isabs(args.outdir):
        raise ValueError('outdir must relative to the current directory')
    outdir_abs = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(outdir_abs):
        raise ValueError("outdir doesn't exist")
    outfile = os.path.join(outdir_abs, os.path.splitext(os.path.split(args.infile)[1])[0])
    process_tempita(args.infile, outfile)
if __name__ == '__main__':
    main()