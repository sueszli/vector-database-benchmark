from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY, version_tuple_to_str
from uncompyle6.scanner import get_scanner

def bug(state, slotstate):
    if False:
        print('Hello World!')
    if state:
        if slotstate is not None:
            for (key, value) in slotstate.items():
                setattr(state, key, 2)

def bug_loop(disassemble, tb=None):
    if False:
        for i in range(10):
            print('nop')
    if tb:
        try:
            tb = 5
        except AttributeError:
            raise RuntimeError
        while tb:
            tb = tb.tb_next
    disassemble(tb)

def test_if_in_for():
    if False:
        i = 10
        return i + 15
    code = bug.__code__
    scan = get_scanner(PYTHON_VERSION_TRIPLE)
    if (2, 7) <= PYTHON_VERSION_TRIPLE < (3, 1) and (not IS_PYPY):
        scan.build_instructions(code)
        fjt = scan.find_jump_targets(False)
        code = bug_loop.__code__
        scan.build_instructions(code)
        fjt = scan.find_jump_targets(False)
        assert {64: [42], 67: [42, 42], 42: [16, 41], 19: [6]} == fjt
        assert scan.structs == [{'start': 0, 'end': 80, 'type': 'root'}, {'start': 3, 'end': 64, 'type': 'if-then'}, {'start': 6, 'end': 15, 'type': 'try'}, {'start': 19, 'end': 38, 'type': 'except'}, {'start': 45, 'end': 67, 'type': 'while-loop'}, {'start': 70, 'end': 64, 'type': 'while-else'}, {'start': 48, 'end': 67, 'type': 'while-loop'}]
    elif (3, 2) < PYTHON_VERSION_TRIPLE <= (3, 4):
        scan.build_instructions(code)
        fjt = scan.find_jump_targets(False)
        assert {69: [66], 63: [18]} == fjt
        assert scan.structs == [{'end': 72, 'type': 'root', 'start': 0}, {'end': 66, 'type': 'if-then', 'start': 6}, {'end': 63, 'type': 'if-then', 'start': 18}, {'end': 59, 'type': 'for-loop', 'start': 31}, {'end': 63, 'type': 'for-else', 'start': 62}]
    else:
        print('FIXME: should fix for %s' % version_tuple_to_str())
        assert True
    return