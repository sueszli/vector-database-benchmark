from synapse.util.stringutils import parse_and_validate_server_name, parse_server_name
from tests import unittest

class ServerNameTestCase(unittest.TestCase):

    def test_parse_server_name(self) -> None:
        if False:
            return 10
        test_data = {'localhost': ('localhost', None), 'my-example.com:1234': ('my-example.com', 1234), '1.2.3.4': ('1.2.3.4', None), '[0abc:1def::1234]': ('[0abc:1def::1234]', None), '1.2.3.4:1': ('1.2.3.4', 1), '[0abc:1def::1234]:8080': ('[0abc:1def::1234]', 8080), ':80': ('', 80), '': ('', None)}
        for (i, o) in test_data.items():
            self.assertEqual(parse_server_name(i), o)

    def test_validate_bad_server_names(self) -> None:
        if False:
            while True:
                i = 10
        test_data = ['', 'localhost:http', '1234]', '[1234', '[1.2.3.4]', 'underscore_.com', 'percent%65.com', 'newline.com\n', '.empty-label.com', '1234:5678:80', ':80']
        for i in test_data:
            try:
                parse_and_validate_server_name(i)
                self.fail("Expected parse_and_validate_server_name('%s') to throw" % (i,))
            except ValueError:
                pass