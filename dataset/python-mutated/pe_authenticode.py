import sys
import lief

def print_crt(binary):
    if False:
        return 10
    for crt in binary.signatures[0].certificates:
        print(crt)
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: {} <pe_binary>'.format(sys.argv[0]))
        sys.exit(1)
    binary = lief.parse(sys.argv[1])
    print_crt(binary)