"""Utility to generate character statistics about a number of source files"""
import argparse
import os

def count_characters(root, out):
    if False:
        while True:
            i = 10
    'Count the occurrances of the different characters in the files'
    if os.path.isfile(root):
        with open(root, 'rb') as in_f:
            for line in in_f:
                for char in line:
                    if char not in out:
                        out[char] = 0
                    out[char] = out[char] + 1
    elif os.path.isdir(root):
        for filename in os.listdir(root):
            count_characters(os.path.join(root, filename), out)

def generate_statistics(root):
    if False:
        return 10
    'Generate the statistics from all files in root (recursively)'
    out = dict()
    count_characters(root, out)
    return out

def main():
    if False:
        print('Hello World!')
    'The main function of the script'
    desc = 'Generate character statistics from a source tree'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--src', dest='src', required=True, help='The root of the source tree')
    parser.add_argument('--out', dest='out', default='chars.py', help='The output filename')
    args = parser.parse_args()
    stats = generate_statistics(args.src)
    with open(args.out, 'wb') as out_f:
        out_f.write('CHARS={0}\n'.format(stats))
if __name__ == '__main__':
    main()