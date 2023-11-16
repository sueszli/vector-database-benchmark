from collections import deque
from xdis import Bytecode, iscode, findlinestarts, get_opcode, offset2line, load_file, load_module

def line_number_mapping(pyc_filename, src_filename):
    if False:
        print('Hello World!')
    (version, timestamp, magic_int, code1, is_pypy, source_size, sip_hash) = load_module(pyc_filename)
    try:
        code2 = load_file(src_filename)
    except SyntaxError as e:
        return str(e)
    queue = deque([code1, code2])
    mappings = []
    opc = get_opcode(version, is_pypy)
    number_loop(queue, mappings, opc)
    return sorted(mappings, key=lambda x: x[1])

def number_loop(queue, mappings, opc):
    if False:
        return 10
    while len(queue) > 0:
        code1 = queue.popleft()
        code2 = queue.popleft()
        assert code1.co_name == code2.co_name
        linestarts_orig = findlinestarts(code1)
        linestarts_uncompiled = list(findlinestarts(code2))
        mappings += [[line, offset2line(offset, linestarts_uncompiled)] for (offset, line) in linestarts_orig]
        bytecode1 = Bytecode(code1, opc)
        bytecode2 = Bytecode(code2, opc)
        instr2s = bytecode2.get_instructions(code2)
        seen = set([code1.co_name])
        for instr in bytecode1.get_instructions(code1):
            next_code1 = None
            if iscode(instr.argval):
                next_code1 = instr.argval
            if next_code1:
                next_code2 = None
                while not next_code2:
                    try:
                        instr2 = next(instr2s)
                        if iscode(instr2.argval):
                            next_code2 = instr2.argval
                            pass
                    except StopIteration:
                        break
                    pass
                if next_code2:
                    assert next_code1.co_name == next_code2.co_name
                    if next_code1.co_name not in seen:
                        seen.add(next_code1.co_name)
                        queue.append(next_code1)
                        queue.append(next_code2)
                        pass
                    pass
            pass
        pass