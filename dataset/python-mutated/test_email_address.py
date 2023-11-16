import unittest
import json
import jc.parsers.email_address

class MyTests(unittest.TestCase):

    def test_email_address_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'email_address' with no data\n        "
        self.assertEqual(jc.parsers.email_address.parse('', quiet=True), {})

    def test_simple_email(self):
        if False:
            return 10
        '\n        Test simple email address\n        '
        data = 'fred@example.com'
        expected = json.loads('{"username":"fred","domain":"example.com","local":"fred","local_plus_suffix":null}')
        self.assertEqual(jc.parsers.email_address.parse(data, quiet=True), expected)

    def test_plus_email(self):
        if False:
            while True:
                i = 10
        '\n        Test email address with plus syntax\n        '
        data = 'fred+spam@example.com'
        expected = json.loads('{"username":"fred","domain":"example.com","local":"fred+spam","local_plus_suffix":"spam"}')
        self.assertEqual(jc.parsers.email_address.parse(data, quiet=True), expected)
if __name__ == '__main__':
    unittest.main()