import pytest
from vyper import ast as vy_ast
from vyper.compiler.phases import CompilerData
from vyper.exceptions import NatSpecSyntaxException
test_code = "\n'''\n@title A simulator for Bug Bunny, the most famous Rabbit\n@license MIT\n@author Warned Bros\n@notice You can use this contract for only the most basic simulation\n@dev\n    Simply chewing a carrot does not count, carrots must pass\n    the throat to be considered eaten\n'''\n\n@external\n@payable\ndef doesEat(food: String[30], qty: uint256) -> bool:\n    '''\n    @notice Determine if Bugs will accept `qty` of `food` to eat\n    @dev Compares the entire string and does not rely on a hash\n    @param food The name of a food to evaluate (in English)\n    @param qty The number of food items to evaluate\n    @return True if Bugs will eat it, False otherwise\n    @custom:my-custom-tag hello, world!\n    '''\n    return True\n"
expected_userdoc = {'methods': {'doesEat(string,uint256)': {'notice': 'Determine if Bugs will accept `qty` of `food` to eat'}}, 'notice': 'You can use this contract for only the most basic simulation'}
expected_devdoc = {'author': 'Warned Bros', 'license': 'MIT', 'details': 'Simply chewing a carrot does not count, carrots must pass the throat to be considered eaten', 'methods': {'doesEat(string,uint256)': {'details': 'Compares the entire string and does not rely on a hash', 'params': {'food': 'The name of a food to evaluate (in English)', 'qty': 'The number of food items to evaluate'}, 'returns': {'_0': 'True if Bugs will eat it, False otherwise'}, 'custom:my-custom-tag': 'hello, world!'}}, 'title': 'A simulator for Bug Bunny, the most famous Rabbit'}

def parse_natspec(code):
    if False:
        return 10
    vyper_ast = CompilerData(code).vyper_module_folded
    return vy_ast.parse_natspec(vyper_ast)

def test_documentation_example_output():
    if False:
        print('Hello World!')
    (userdoc, devdoc) = parse_natspec(test_code)
    assert userdoc == expected_userdoc
    assert devdoc == expected_devdoc

def test_no_tags_implies_notice():
    if False:
        print('Hello World!')
    code = "\n'''\nBecause there is no tag, this docstring is handled as a notice.\n'''\n@external\ndef foo():\n    '''\n    This one too!\n    '''\n    pass\n    "
    (userdoc, devdoc) = parse_natspec(code)
    assert userdoc == {'methods': {'foo()': {'notice': 'This one too!'}}, 'notice': 'Because there is no tag, this docstring is handled as a notice.'}
    assert not devdoc

def test_whitespace():
    if False:
        print('Hello World!')
    code = "\n'''\n        @dev\n\n  Whitespace    gets  cleaned\n    up,\n            people can use\n\n\n         awful formatting.\n\n\nWe don't mind!\n\n@author Mr No-linter\n                '''\n"
    (_, devdoc) = parse_natspec(code)
    assert devdoc == {'author': 'Mr No-linter', 'details': "Whitespace gets cleaned up, people can use awful formatting. We don't mind!"}

def test_params():
    if False:
        while True:
            i = 10
    code = "\n@external\ndef foo(bar: int128, baz: uint256, potato: bytes32):\n    '''\n    @param bar a number\n    @param baz also a number\n    @dev we didn't document potato, but that's ok\n    '''\n    pass\n    "
    (_, devdoc) = parse_natspec(code)
    assert devdoc == {'methods': {'foo(int128,uint256,bytes32)': {'details': "we didn't document potato, but that's ok", 'params': {'bar': 'a number', 'baz': 'also a number'}}}}

def test_returns():
    if False:
        return 10
    code = "\n@external\ndef foo(bar: int128, baz: uint256) -> (int128, uint256):\n    '''\n    @return value of bar\n    @return value of baz\n    '''\n    return bar, baz\n    "
    (_, devdoc) = parse_natspec(code)
    assert devdoc == {'methods': {'foo(int128,uint256)': {'returns': {'_0': 'value of bar', '_1': 'value of baz'}}}}

def test_ignore_private_methods():
    if False:
        i = 10
        return i + 15
    code = "\n@external\ndef foo(bar: int128, baz: uint256):\n    '''@dev I will be parsed.'''\n    pass\n\n@internal\ndef notfoo(bar: int128, baz: uint256):\n    '''@dev I will not be parsed.'''\n    pass\n    "
    (_, devdoc) = parse_natspec(code)
    assert devdoc['methods'] == {'foo(int128,uint256)': {'details': 'I will be parsed.'}}

def test_partial_natspec():
    if False:
        while True:
            i = 10
    code = "\n@external\ndef foo():\n    '''\n    Regular comments preceeding natspec is not allowed\n    @notice this is natspec\n    '''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match='NatSpec docstring opens with untagged comment'):
        parse_natspec(code)
empty_field_cases = ["\n    @notice\n    @dev notice shouldn't be empty\n    @author nobody\n    ", "\n    @dev notice shouldn't be empty\n    @notice\n    @author nobody\n    ", "\n    @dev notice shouldn't be empty\n    @author nobody\n    @notice\n    "]

@pytest.mark.parametrize('bad_docstring', empty_field_cases)
def test_empty_field(bad_docstring):
    if False:
        return 10
    code = f"\n@external\ndef foo():\n    '''{bad_docstring}'''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="No description given for tag '@notice'"):
        parse_natspec(code)

def test_unknown_field():
    if False:
        while True:
            i = 10
    code = "\n@external\ndef foo():\n    '''\n    @notice this is ok\n    @thing this is bad\n    '''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="Unknown NatSpec field '@thing'"):
        parse_natspec(code)

@pytest.mark.parametrize('field', ['title', 'license'])
def test_invalid_field(field):
    if False:
        print('Hello World!')
    code = f"\n@external\ndef foo():\n    '''@{field} function level docstrings cannot have titles'''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match=f"'@{field}' is not a valid field"):
        parse_natspec(code)
licenses = ['Apache-2.0', 'Apache-2.0 OR MIT', 'PSF-2.0 AND MIT', 'Apache-2.0 AND (MIT OR GPL-2.0-only)']

@pytest.mark.parametrize('license', licenses)
def test_license(license):
    if False:
        return 10
    code = f"\n'''\n@license {license}\n'''\n@external\ndef foo():\n    pass\n    "
    (_, devdoc) = parse_natspec(code)
    assert devdoc == {'license': license}
fields = ['title', 'author', 'license', 'notice', 'dev']

@pytest.mark.parametrize('field', fields)
def test_empty_fields(field):
    if False:
        return 10
    code = f"\n'''\n@{field}\n'''\n@external\ndef foo():\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match=f"No description given for tag '@{field}'"):
        parse_natspec(code)

def test_duplicate_fields():
    if False:
        print('Hello World!')
    code = "\n@external\ndef foo():\n    '''\n    @notice It's fine to have one notice, but....\n    @notice a second one, not so much\n    '''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="Duplicate NatSpec field '@notice'"):
        parse_natspec(code)

def test_duplicate_param():
    if False:
        i = 10
        return i + 15
    code = "\n@external\ndef foo(bar: int128, baz: uint256):\n    '''\n    @param bar a number\n    @param bar also a number\n    '''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="Parameter 'bar' documented more than once"):
        parse_natspec(code)

def test_unknown_param():
    if False:
        i = 10
        return i + 15
    code = "\n@external\ndef foo(bar: int128, baz: uint256):\n    '''@param hotdog not a number'''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="Method has no parameter 'hotdog'"):
        parse_natspec(code)
empty_field_cases = ["\n    @param a\n    @dev param shouldn't be empty\n    @author nobody\n    ", "\n    @dev param shouldn't be empty\n    @param a\n    @author nobody\n    ", "\n    @dev param shouldn't be empty\n    @author nobody\n    @param a\n    "]

@pytest.mark.parametrize('bad_docstring', empty_field_cases)
def test_empty_param(bad_docstring):
    if False:
        i = 10
        return i + 15
    code = f"\n@external\ndef foo(a: int128):\n    '''{bad_docstring}'''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match="No description given for parameter 'a'"):
        parse_natspec(code)

def test_too_many_returns_no_return_type():
    if False:
        return 10
    code = "\n@external\ndef foo():\n    '''@return should fail, the function does not include a return value'''\n    pass\n    "
    with pytest.raises(NatSpecSyntaxException, match='Method does not return any values'):
        parse_natspec(code)

def test_too_many_returns_single_return_type():
    if False:
        i = 10
        return i + 15
    code = "\n@external\ndef foo() -> int128:\n    '''\n    @return int128\n    @return this should fail\n    '''\n    return 1\n    "
    with pytest.raises(NatSpecSyntaxException, match='Number of documented return values exceeds actual number'):
        parse_natspec(code)

def test_too_many_returns_tuple_return_type():
    if False:
        for i in range(10):
            print('nop')
    code = "\n@external\ndef foo() -> (int128,uint256):\n    '''\n    @return int128\n    @return uint256\n    @return this should fail\n    '''\n    return 1, 2\n    "
    with pytest.raises(NatSpecSyntaxException, match='Number of documented return values exceeds actual number'):
        parse_natspec(code)