from salt.utils.sanitizers import clean, mask_args_value
from tests.support.unit import TestCase

class SanitizersTestCase(TestCase):
    """
    TestCase for sanitizers
    """

    def test_sanitized_trim(self):
        if False:
            i = 10
            return i + 15
        '\n        Test sanitized input for trimming\n        '
        value = ' sample '
        response = clean.trim(value)
        assert response == 'sample'
        assert type(response) == str

    def test_sanitized_filename(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test sanitized input for filename\n        '
        value = '/absolute/path/to/the/file.txt'
        response = clean.filename(value)
        assert response == 'file.txt'
        value = '../relative/path/to/the/file.txt'
        response = clean.filename(value)
        assert response == 'file.txt'

    def test_sanitized_hostname(self):
        if False:
            return 10
        '\n        Test sanitized input for hostname (id)\n        '
        value = '   ../ ../some/dubious/hostname      '
        response = clean.hostname(value)
        assert response == 'somedubioushostname'
    test_sanitized_id = test_sanitized_hostname

    def test_value_masked(self):
        if False:
            print('Hello World!')
        '\n        Test if the values are masked.\n        :return:\n        '
        out = mask_args_value('quantum: fluctuations', 'quant*')
        assert out == 'quantum: ** hidden **'

    def test_value_not_masked(self):
        if False:
            return 10
        '\n        Test if the values are not masked.\n        :return:\n        '
        out = mask_args_value('quantum fluctuations', 'quant*')
        assert out == 'quantum fluctuations'