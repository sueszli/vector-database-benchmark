from unittest.case import skip, skipIf

def generate_skip_decorator(bear):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a skip decorator for a `unittest` module test from a bear.\n\n    `check_prerequisites` is used to determine a test skip.\n\n    :param bear: The bear whose prerequisites determine the test skip.\n    :return:     A decorator that skips the test if appropriate.\n    '
    result = bear.check_prerequisites()
    return skip(result) if isinstance(result, str) else skipIf(not result, '(No reason given.)')