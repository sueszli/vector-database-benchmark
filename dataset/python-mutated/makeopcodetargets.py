"""Generate C code for the jump table of the threaded code interpreter
(for compilers supporting computed gotos or "labels-as-values", such as gcc).
"""
import os
import sys
try:
    from importlib.machinery import SourceFileLoader
except ImportError:
    import imp

    def find_module(modname):
        if False:
            while True:
                i = 10
        'Finds and returns a module in the local dist/checkout.\n        '
        modpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Lib')
        return imp.load_module(modname, *imp.find_module(modname, [modpath]))
else:

    def find_module(modname):
        if False:
            return 10
        'Finds and returns a module in the local dist/checkout.\n        '
        modpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Lib', modname + '.py')
        return SourceFileLoader(modname, modpath).load_module()

def write_contents(f):
    if False:
        return 10
    'Write C code contents to the target file object.\n    '
    opcode = find_module('opcode')
    targets = [None] * 256
    for (opname, op) in opcode.opmap.items():
        targets[op] = opname
    f.write('static void *opcode_targets[256] = {\n')
    for s in targets:
        if s is None:
            f.write(f'    &&_unknown_opcode,\n')
            continue
        if s in opcode.cinderxop:
            f.write(f'#ifdef ENABLE_CINDERX\n')
        f.write(f'    &&TARGET_{s},\n')
        if s in opcode.cinderxop:
            f.write(f'#else\n    &&_unknown_opcode,\n#endif\n')
    f.write('\n};\n')

def main():
    if False:
        i = 10
        return i + 15
    if len(sys.argv) >= 3:
        sys.exit('Too many arguments')
    if len(sys.argv) == 2:
        target = sys.argv[1]
    else:
        target = 'Python/opcode_targets.h'
    with open(target, 'w') as f:
        write_contents(f)
    print('Jump table written into %s' % target)
if __name__ == '__main__':
    main()