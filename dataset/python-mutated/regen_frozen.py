import sys
import os
import marshal
DIR = os.path.dirname(sys.argv[0])
FILE = os.path.join(DIR, 'flag.py')
SYMBOL = 'M___hello__'

def get_module_code(filename):
    if False:
        print('Hello World!')
    "Compile 'filename' and return the module code as a marshalled byte\n    string.\n    "
    with open(filename, 'r') as fp:
        src = fp.read()
    co = compile(src, 'none', 'exec')
    co_bytes = marshal.dumps(co)
    return co_bytes

def gen_c_code(fp, co_bytes):
    if False:
        print('Hello World!')
    "Generate C code for the module code in 'co_bytes', write it to 'fp'.\n    "

    def write(*args, **kwargs):
        if False:
            print('Hello World!')
        print(*args, **kwargs, file=fp)
    write('/* Generated with Tools/freeze/regen_frozen.py */')
    write('static unsigned char %s[] = {' % SYMBOL, end='')
    bytes_per_row = 13
    for (i, opcode) in enumerate(co_bytes):
        if i % bytes_per_row == 0:
            write()
            write('    ', end='')
        write('%d,' % opcode, end='')
    write()
    write('};')

def main():
    if False:
        return 10
    out_filename = sys.argv[1]
    co_bytes = get_module_code(FILE)
    with open(out_filename, 'w') as fp:
        gen_c_code(fp, co_bytes)
if __name__ == '__main__':
    main()