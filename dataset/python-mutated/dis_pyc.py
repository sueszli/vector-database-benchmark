import argparse
import dis
import marshal
import sys
from types import CodeType
try:
    from compiler.dis_stable import Disassembler
except ImportError:
    from dis import dis
else:

    def dis(code, file=None):
        if False:
            i = 10
            return i + 15
        Disassembler().dump_code(code, file=sys.stdout)

def _get_code(root, item_path=None):
    if False:
        i = 10
        return i + 15
    if not item_path:
        return root
    code = root
    for chunk in item_path.split('.'):
        for subcode in code.co_consts:
            if isinstance(subcode, CodeType):
                if subcode.co_name == chunk:
                    code = subcode
                    break
        else:
            print(f'Could not find code object for "{chunk}" in "{item_path}"')
            sys.exit(1)
    return code

def main(pyc_path, item_path=None):
    if False:
        while True:
            i = 10
    with open(sys.argv[1], 'rb') as f:
        f.seek(20)
        dis(_get_code(marshal.load(f), item_path), file=sys.stdout)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prints the disassembly of bytecode contained in a pyc file')
    parser.add_argument('pyc_path', help='Path to the pyc file')
    parser.add_argument('--itempath', help='Only disassemble this code object within the module. e.g: ClassName.method')
    args = parser.parse_args()
    print(args.itempath)
    main(args.pyc_path, args.itempath)