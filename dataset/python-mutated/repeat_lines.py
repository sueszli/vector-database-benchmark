import argparse
import sys

def _normalize_spaces(line):
    if False:
        i = 10
        return i + 15
    return ' '.join(line.split())

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('-n', '--repeat_times', required=True, type=int)
    parser.add_argument('-o', '--output_file', required=False, type=str)
    args = parser.parse_args()
    stream = open(args.output_file, 'w') if args.output_file else sys.stdout
    for line in open(args.input_file):
        for _ in range(args.repeat_times):
            stream.write(_normalize_spaces(line) + '\n')
if __name__ == '__main__':
    main()