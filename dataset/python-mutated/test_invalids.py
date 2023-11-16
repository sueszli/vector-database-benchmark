import pytest
from vyper import compiler
from vyper.exceptions import FunctionDeclarationException, InvalidOperation, InvalidType, StructureException, TypeMismatch, UndeclaredDefinition, UnknownAttribute
fail_list = []

def must_fail(code, exception):
    if False:
        print('Hello World!')
    fail_list.append((code, exception))
pass_list = []

def must_succeed(code):
    if False:
        for i in range(10):
            print('nop')
    pass_list.append(code)
must_succeed('\nx: int128[3]\n')
must_succeed('\n@external\ndef foo(x: int128): pass\n')
must_succeed('\n@external\ndef foo():\n    x: int128 = 0\n    x = 5\n')
must_succeed('\n@external\ndef foo():\n    x: int128  = 5\n')
must_fail('\n@external\ndef foo():\n    x: int128 = 5\n    x = 0x1234567890123456789012345678901234567890\n', InvalidType)
must_fail('\n@external\ndef foo():\n    x: int128 = 5\n    x = 3.5\n', InvalidType)
must_succeed('\n@external\ndef foo():\n    x: int128 = 5\n    x = 3\n')
must_succeed('\nb: int128\n@external\ndef foo():\n    self.b = 7\n')
must_fail('\nb: int128\n@external\ndef foo():\n    self.b = 7.5\n', InvalidType)
must_succeed('\nb: decimal\n@external\ndef foo():\n    self.b = 7.5\n')
must_succeed('\nb: decimal\n@external\ndef foo():\n    self.b = 7.0\n')
must_fail('\nb: int128[5]\n@external\ndef foo():\n    self.b = 7\n', InvalidType)
must_succeed('\nb: HashMap[int128, int128]\n@external\ndef foo():\n    x: int128 = self.b[5]\n')
must_fail('\nb: HashMap[uint256, uint256]\n@external\ndef foo():\n    x: int128 = self.b[-5]\n', InvalidType)
must_fail('\nb: HashMap[int128, int128]\n@external\ndef foo():\n    x: int128 = self.b[5.7]\n', InvalidType)
must_succeed('\nb: HashMap[decimal, int128]\n@external\ndef foo():\n    x: int128 = self.b[5.0]\n')
must_fail('\nb: HashMap[int128, int128]\n@external\ndef foo():\n    self.b[3] = 5.6\n', InvalidType)
must_succeed('\nb: HashMap[int128, int128]\n@external\ndef foo():\n    self.b[3] = -5\n')
must_succeed('\nb: HashMap[int128, int128]\n@external\ndef foo():\n    self.b[-3] = 5\n')
must_succeed('\n@external\ndef foo():\n    x: int128[5] = [0, 0, 0, 0, 0]\n    z: int128 = x[2]\n')
must_succeed('\nx: int128\n@external\ndef foo():\n    self.x = 5\n')
must_succeed('\nx: int128\n@internal\ndef foo():\n    self.x = 5\n')
must_fail('\nbar: int128[3]\n@external\ndef foo():\n    self.bar = 5\n', InvalidType)
must_succeed('\nbar: int128[3]\n@external\ndef foo():\n    self.bar[0] = 5\n')
must_fail('\n@external\ndef foo() -> address:\n    return [1, 2, 3]\n', InvalidType)
must_fail('\n@external\ndef baa() -> decimal:\n    return 2.0**2\n', TypeMismatch)
must_succeed('\n@external\ndef foo():\n    raise "fail"\n')
must_succeed('\n@internal\ndef foo():\n    pass\n\n@external\ndef goo():\n    self.foo()\n')
must_succeed('\n@external\ndef foo():\n    MOOSE: int128 = 45\n')
must_fail('\n@external\ndef foo():\n    x: address = -self\n', InvalidOperation)
must_fail('\n@external\ndef foo() -> int128:\n    return\n', FunctionDeclarationException)
must_fail('\n@external\ndef foo():\n    return 3\n', FunctionDeclarationException)
must_fail('\n@external\ndef foo():\n    suicide(msg.sender)\n    ', UndeclaredDefinition)
must_succeed('\n@external\ndef sum(a: int128, b: int128) -> int128:\n    """\n    Sum two signed integers.\n    """\n    return a + b\n')
must_fail('\n@external\ndef a():\n    "Behold me mortal, for I am a DOCSTRING!"\n    "Alas, I am but a mere string."\n', StructureException)
must_fail('\nstruct StructX:\n    x: int128\n\n@external\ndef a():\n    x: int128 = StructX({y: 1})\n', UnknownAttribute)
must_fail('\na: HashMap\n', StructureException)

@pytest.mark.parametrize('bad_code,exception_type', fail_list)
def test_compilation_fails_with_exception(bad_code, exception_type):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(exception_type):
        compiler.compile_code(bad_code)

@pytest.mark.parametrize('good_code', pass_list)
def test_compilation_succeeds(good_code):
    if False:
        print('Hello World!')
    assert compiler.compile_code(good_code) is not None