import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import keys

def check_icons(actual):
    if False:
        return 10
    expected = '[START icons]\n🍓\n🥕\n🍆\n🍅\n🥔\n[END icons]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.keys.print', str)
class KeysTest(unittest.TestCase):

    def test_keys(self):
        if False:
            for i in range(10):
                print('nop')
        keys.keys(check_icons)
if __name__ == '__main__':
    unittest.main()