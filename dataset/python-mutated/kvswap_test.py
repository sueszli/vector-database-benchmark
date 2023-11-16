import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import kvswap

def check_plants(actual):
    if False:
        print('Hello World!')
    expected = "[START plants]\n('Strawberry', '🍓')\n('Carrot', '🥕')\n('Eggplant', '🍆')\n('Tomato', '🍅')\n('Potato', '🥔')\n[END plants]".splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.kvswap.print', str)
class KvSwapTest(unittest.TestCase):

    def test_kvswap(self):
        if False:
            while True:
                i = 10
        kvswap.kvswap(check_plants)
if __name__ == '__main__':
    unittest.main()