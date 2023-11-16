"""Test pycode.ast"""
import ast
import pytest
from sphinx.pycode.ast import unparse as ast_unparse

@pytest.mark.parametrize(('source', 'expected'), [('a + b', 'a + b'), ('a and b', 'a and b'), ('os.path', 'os.path'), ('1 * 2', '1 * 2'), ('a & b', 'a & b'), ('a | b', 'a | b'), ('a ^ b', 'a ^ b'), ('a and b and c', 'a and b and c'), ("b'bytes'", "b'bytes'"), ('object()', 'object()'), ('1234', '1234'), ("{'key1': 'value1', 'key2': 'value2'}", "{'key1': 'value1', 'key2': 'value2'}"), ('a / b', 'a / b'), ('...', '...'), ('a // b', 'a // b'), ('Tuple[int, int]', 'Tuple[int, int]'), ('~1', '~1'), ('lambda x, y: x + y', 'lambda x, y: ...'), ('[1, 2, 3]', '[1, 2, 3]'), ('a << b', 'a << b'), ('a @ b', 'a @ b'), ('a % b', 'a % b'), ('a * b', 'a * b'), ('sys', 'sys'), ('1234', '1234'), ('not a', 'not a'), ('a or b', 'a or b'), ('a**b', 'a**b'), ('a >> b', 'a >> b'), ('{1, 2, 3}', '{1, 2, 3}'), ('a - b', 'a - b'), ("'str'", "'str'"), ('+a', '+a'), ('-1', '-1'), ('-a', '-a'), ('(1, 2, 3)', '(1, 2, 3)'), ('()', '()'), ('(1,)', '(1,)'), ('lambda x=0, /, y=1, *args, z, **kwargs: x + y + z', 'lambda x=0, /, y=1, *args, z, **kwargs: ...'), ('0x1234', '0x1234'), ('1_000_000', '1_000_000')])
def test_unparse(source, expected):
    if False:
        while True:
            i = 10
    module = ast.parse(source)
    assert ast_unparse(module.body[0].value, source) == expected

def test_unparse_None():
    if False:
        i = 10
        return i + 15
    assert ast_unparse(None) is None