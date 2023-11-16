import argparse
import fileinput
import sacremoses

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()
    detok = sacremoses.MosesDetokenizer()
    for line in fileinput.input(args.files, openhook=fileinput.hook_compressed):
        print(detok.detokenize(line.strip().split(' ')).replace(' @', '').replace('@ ', '').replace(' =', '=').replace('= ', '=').replace(' – ', '–'))
if __name__ == '__main__':
    main()