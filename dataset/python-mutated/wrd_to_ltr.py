import sys

def main():
    if False:
        return 10
    for line in sys.stdin:
        print(' '.join(list(line.strip().replace(' ', '|'))) + ' |')
if __name__ == '__main__':
    main()