import sys
from lief import parse

def nm(filename):
    if False:
        print('Hello World!')
    ' Return symbols from *filename* binary '
    binary = parse(filename)
    symbols = binary.symbols
    if len(symbols) > 0:
        for symbol in symbols:
            print(symbol)
    else:
        print('No symbols found')
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: ' + sys.argv[0] + ' <binary>')
        sys.exit(-1)
    nm(sys.argv[1])