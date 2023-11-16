import pytest
from scripts.validate_exception_location import ERROR_MESSAGE, validate_exception_and_warning_placement
PATH = 't.py'
CUSTOM_EXCEPTION_NOT_IN_TESTING_RST = 'MyException'
CUSTOM_EXCEPTION__IN_TESTING_RST = 'MyOldException'
ERRORS_IN_TESTING_RST = {CUSTOM_EXCEPTION__IN_TESTING_RST}
TEST_CODE = '\nimport numpy as np\nimport sys\n\ndef my_func():\n  pass\n\nclass {custom_name}({error_type}):\n  pass\n\n'

@pytest.fixture(params=['Exception', 'ValueError', 'Warning', 'UserWarning'])
def error_type(request):
    if False:
        print('Hello World!')
    return request.param

def test_class_that_inherits_an_exception_and_is_not_in_the_testing_rst_is_flagged(capsys, error_type):
    if False:
        print('Hello World!')
    content = TEST_CODE.format(custom_name=CUSTOM_EXCEPTION_NOT_IN_TESTING_RST, error_type=error_type)
    expected_msg = ERROR_MESSAGE.format(errors=CUSTOM_EXCEPTION_NOT_IN_TESTING_RST)
    with pytest.raises(SystemExit, match=None):
        validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)
    (result_msg, _) = capsys.readouterr()
    assert result_msg == expected_msg

def test_class_that_inherits_an_exception_but_is_in_the_testing_rst_is_not_flagged(capsys, error_type):
    if False:
        i = 10
        return i + 15
    content = TEST_CODE.format(custom_name=CUSTOM_EXCEPTION__IN_TESTING_RST, error_type=error_type)
    validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)

def test_class_that_does_not_inherit_an_exception_is_not_flagged(capsys):
    if False:
        for i in range(10):
            print('nop')
    content = 'class MyClass(NonExceptionClass): pass'
    validate_exception_and_warning_placement(PATH, content, ERRORS_IN_TESTING_RST)