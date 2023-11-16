import lief
import sys
import random, string

def randomword(length):
    if False:
        return 10
    return ''.join((random.choice(string.ascii_lowercase) for i in range(length)))

def randomize(binary, output):
    if False:
        i = 10
        return i + 15
    symbols = binary.static_symbols
    if len(symbols) == 0:
        print('No symbols')
        return
    for symbol in symbols:
        symbol.name = randomword(len(symbol.name))
    binary.write(output)

def main():
    if False:
        while True:
            i = 10
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], '<elf binary> <output binary>')
        sys.exit(-1)
    binary = lief.parse(sys.argv[1])
    randomize(binary, sys.argv[2])
if __name__ == '__main__':
    main()