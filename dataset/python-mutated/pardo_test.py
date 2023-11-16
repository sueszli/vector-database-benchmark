import unittest
from io import StringIO
import mock
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from . import pardo_dofn
from . import pardo_dofn_methods
from . import pardo_dofn_params

def check_plants(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '[START plants]\n🍓Strawberry\n🥕Carrot\n🍆Eggplant\n🍅Tomato\n🥔Potato\n[END plants]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_dofn_params(actual):
    if False:
        for i in range(10):
            print('nop')
    expected = '\n'.join("[START dofn_params]\n# timestamp\ntype(timestamp) -> <class 'apache_beam.utils.timestamp.Timestamp'>\ntimestamp.micros -> 1584675660000000\ntimestamp.to_rfc3339() -> '2020-03-20T03:41:00Z'\ntimestamp.to_utc_datetime() -> datetime.datetime(2020, 3, 20, 3, 41)\n\n# window\ntype(window) -> <class 'apache_beam.transforms.window.IntervalWindow'>\nwindow.start -> Timestamp(1584675660) (2020-03-20 03:41:00)\nwindow.end -> Timestamp(1584675690) (2020-03-20 03:41:30)\nwindow.max_timestamp() -> Timestamp(1584675689.999999) (2020-03-20 03:41:29.999999)\n[END dofn_params]".splitlines()[1:-1])
    assert_that(actual, equal_to([expected]))

def check_dofn_methods(actual):
    if False:
        i = 10
        return i + 15
    return '[START results]\n__init__\nsetup\nstart_bundle\n* process: 🍓\n* process: 🥕\n* process: 🍆\n* process: 🍅\n* process: 🥔\n* finish_bundle: 🌱🌳🌍\nteardown\n[END results]'.splitlines()[1:-1]

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.pardo_dofn.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.pardo_dofn_params.print', str)
class ParDoTest(unittest.TestCase):

    def test_pardo_dofn(self):
        if False:
            for i in range(10):
                print('nop')
        pardo_dofn.pardo_dofn(check_plants)

    def test_pardo_dofn_params(self):
        if False:
            while True:
                i = 10
        pardo_dofn_params.pardo_dofn_params(check_dofn_params)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('sys.stdout', new_callable=StringIO)
class ParDoStdoutTest(unittest.TestCase):

    def test_pardo_dofn_methods(self, mock_stdout):
        if False:
            i = 10
            return i + 15
        expected = pardo_dofn_methods.pardo_dofn_methods(check_dofn_methods)
        actual = mock_stdout.getvalue().splitlines()
        actual_stdout = [line.split(':')[0] for line in actual]
        expected_stdout = [line.split(':')[0] for line in expected]
        self.assertEqual(actual_stdout, expected_stdout)
        actual_elements = {line for line in actual if line.startswith('*')}
        expected_elements = {line for line in expected if line.startswith('*')}
        self.assertEqual(actual_elements, expected_elements)
if __name__ == '__main__':
    unittest.main()