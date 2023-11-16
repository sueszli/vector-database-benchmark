import unittest
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import cogroupbykey

def check_plants(actual):
    if False:
        print('Hello World!')
    expected = "[START plants]\n('Apple', {'icons': ['🍎', '🍏'], 'durations': ['perennial']})\n('Carrot', {'icons': [], 'durations': ['biennial']})\n('Tomato', {'icons': ['🍅'], 'durations': ['perennial', 'annual']})\n('Eggplant', {'icons': ['🍆'], 'durations': []})\n[END plants]".splitlines()[1:-1]

    def normalize_element(elem):
        if False:
            return 10
        (name, details) = elem
        details['icons'] = sorted(details['icons'])
        details['durations'] = sorted(details['durations'])
        return (name, details)
    assert_matches_stdout(actual, expected, normalize_element)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.aggregation.cogroupbykey.print', str)
class CoGroupByKeyTest(unittest.TestCase):

    def test_cogroupbykey(self):
        if False:
            for i in range(10):
                print('nop')
        cogroupbykey.cogroupbykey(check_plants)
if __name__ == '__main__':
    unittest.main()