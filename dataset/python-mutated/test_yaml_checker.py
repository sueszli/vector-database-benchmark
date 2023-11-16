import yaml
from grc.core.schema_checker import Validator, BLOCK_SCHEME
BLOCK1 = "\nid: block_key\nlabel: testname\n\nparameters:\n-   id: vlen\n    label: Vec Length\n    dtype: int\n    default: 1\n-   id: out_type\n    label: Vec Length\n    dtype: string\n    default: complex\n-   id: a\n    label: Alpha\n    dtype: ${ out_type }\n    default: '0'\n\ninputs:\n-   label: in\n    domain: stream\n    dtype: complex\n    vlen: ${ 2 * vlen }\n-   name: in2\n    domain: message\n    id: in2\n\noutputs:\n-   label: out\n    domain: stream\n    dtype: ${ out_type }\n    vlen: ${ vlen }\n\ntemplates:\n    make: blocks.complex_to_mag_squared(${ vlen })\n\nfile_format: 1\n"

def test_min():
    if False:
        i = 10
        return i + 15
    checker = Validator(BLOCK_SCHEME)
    assert checker.run({'id': 'test', 'file_format': 1}), checker.messages
    assert not checker.run({'name': 'test', 'file_format': 1})

def test_extra_keys():
    if False:
        return 10
    checker = Validator(BLOCK_SCHEME)
    assert checker.run({'id': 'test', 'abcdefg': 'nonsense', 'file_format': 1})
    assert checker.messages == [('block', 'warn', "Ignoring extra key 'abcdefg'")]

def test_checker():
    if False:
        print('Hello World!')
    checker = Validator(BLOCK_SCHEME)
    data = yaml.safe_load(BLOCK1)
    passed = checker.run(data)
    if not passed:
        print()
        for msg in checker.messages:
            print(msg)
    assert passed, checker.messages