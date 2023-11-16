import regex
import sys

def main():
    if False:
        return 10
    filter_r = regex.compile("[^\\p{L}\\p{N}\\p{M}\\' \\-]")
    for line in sys.stdin:
        line = line.strip()
        line = filter_r.sub(' ', line)
        line = ' '.join(line.split())
        print(line)
if __name__ == '__main__':
    main()