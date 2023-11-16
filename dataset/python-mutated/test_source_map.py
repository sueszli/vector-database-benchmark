from vyper.compiler import compile_code
from vyper.compiler.output import _compress_source_map
from vyper.compiler.utils import expand_source_map
TEST_CODE = '\n@internal\ndef _baz(a: int128) -> int128:\n    b: int128 = a\n    for i in range(2, 5):\n        b *=  i\n        if b > 31337:\n            break\n    return b\n\n@internal\ndef _bar(a: uint256) -> bool:\n    if a > 42:\n        return True\n    return False\n\n@external\ndef foo(a: uint256) -> int128:\n    if self._bar(a):\n        return self._baz(2)\n    else:\n        return 42\n    '

def test_jump_map():
    if False:
        while True:
            i = 10
    source_map = compile_code(TEST_CODE, output_formats=['source_map'])['source_map']
    pos_map = source_map['pc_pos_map']
    jump_map = source_map['pc_jump_map']
    assert len([v for v in jump_map.values() if v == 'o']) == 1
    assert len([v for v in jump_map.values() if v == 'i']) == 2
    code_lines = [i + '\n' for i in TEST_CODE.split('\n')]
    for pc in [k for (k, v) in jump_map.items() if v == 'o']:
        (lineno, col_offset, _, end_col_offset) = pos_map[pc]
        assert code_lines[lineno - 1][col_offset:end_col_offset].startswith('return')
    for pc in [k for (k, v) in jump_map.items() if v == 'i']:
        (lineno, col_offset, _, end_col_offset) = pos_map[pc]
        assert code_lines[lineno - 1][col_offset:end_col_offset].startswith('self.')

def test_pos_map_offsets():
    if False:
        return 10
    source_map = compile_code(TEST_CODE, output_formats=['source_map'])['source_map']
    expanded = expand_source_map(source_map['pc_pos_map_compressed'])
    pc_iter = iter((source_map['pc_pos_map'][i] for i in sorted(source_map['pc_pos_map'])))
    jump_iter = iter((source_map['pc_jump_map'][i] for i in sorted(source_map['pc_jump_map'])))
    code_lines = [i + '\n' for i in TEST_CODE.split('\n')]
    for item in expanded:
        if item[-1] is not None:
            assert next(jump_iter) == item[-1]
        if item[:2] != [-1, -1]:
            (start, length) = item[:2]
            (lineno, col_offset, end_lineno, end_col_offset) = next(pc_iter)
            assert code_lines[lineno - 1][col_offset] == TEST_CODE[start]
            assert length == sum((len(i) for i in code_lines[lineno - 1:end_lineno])) - col_offset - (len(code_lines[end_lineno - 1]) - end_col_offset)

def test_error_map():
    if False:
        for i in range(10):
            print('nop')
    code = '\nfoo: uint256\n\n@external\ndef update_foo():\n    self.foo += 1\n    '
    error_map = compile_code(code, output_formats=['source_map'])['source_map']['error_map']
    assert 'safeadd' in list(error_map.values())
    assert 'fallback function' in list(error_map.values())

def test_compress_source_map():
    if False:
        i = 10
        return i + 15
    code = '\n@external\ndef foo() -> uint256:\n    return 42\n    '
    compressed = _compress_source_map(code, {'0': None, '2': (2, 0, 4, 13), '3': (2, 0, 2, 8), '5': (2, 0, 2, 8)}, {'3': 'o'}, 2)
    assert compressed == '-1:-1:2:-;1:45;:8::o;'

def test_expand_source_map():
    if False:
        for i in range(10):
            print('nop')
    compressed = '-1:-1:0:-;;13:42:1;:21;::0:o;:::-;1::1;'
    expanded = [[-1, -1, 0, '-'], [-1, -1, 0, None], [13, 42, 1, None], [13, 21, 1, None], [13, 21, 0, 'o'], [13, 21, 0, '-'], [1, 21, 1, None]]
    assert expand_source_map(compressed) == expanded