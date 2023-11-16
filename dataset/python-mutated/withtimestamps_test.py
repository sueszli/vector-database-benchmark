import unittest
import mock
import apache_beam as beam
from apache_beam.examples.snippets.util import assert_matches_stdout
from apache_beam.testing.test_pipeline import TestPipeline
from . import withtimestamps
from . import withtimestamps_event_time
from . import withtimestamps_logical_clock
from . import withtimestamps_processing_time

def check_plant_timestamps(actual):
    if False:
        while True:
            i = 10
    expected = '[START plant_timestamps]\n2020-04-01 00:00:00 - Strawberry\n2020-06-01 00:00:00 - Carrot\n2020-03-01 00:00:00 - Artichoke\n2020-05-01 00:00:00 - Tomato\n2020-09-01 00:00:00 - Potato\n[END plant_timestamps]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_plant_events(actual):
    if False:
        print('Hello World!')
    expected = '[START plant_events]\n1 - Strawberry\n4 - Carrot\n2 - Artichoke\n3 - Tomato\n5 - Potato\n[END plant_events]'.splitlines()[1:-1]
    assert_matches_stdout(actual, expected)

def check_plant_processing_times(actual):
    if False:
        i = 10
        return i + 15
    expected = '[START plant_processing_times]\n2020-03-20 20:12:42.145594 - Strawberry\n2020-03-20 20:12:42.145827 - Carrot\n2020-03-20 20:12:42.145962 - Artichoke\n2020-03-20 20:12:42.146093 - Tomato\n2020-03-20 20:12:42.146216 - Potato\n[END plant_processing_times]'.splitlines()[1:-1]
    actual = actual | beam.Map(lambda row: row.split('-')[-1].strip())
    expected = [row.split('-')[-1].strip() for row in expected]
    assert_matches_stdout(actual, expected)

@mock.patch('apache_beam.Pipeline', TestPipeline)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.withtimestamps_event_time.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.withtimestamps_logical_clock.print', str)
@mock.patch('apache_beam.examples.snippets.transforms.elementwise.withtimestamps_processing_time.print', str)
class WithTimestampsTest(unittest.TestCase):

    def test_event_time(self):
        if False:
            return 10
        withtimestamps_event_time.withtimestamps_event_time(check_plant_timestamps)

    def test_logical_clock(self):
        if False:
            while True:
                i = 10
        withtimestamps_logical_clock.withtimestamps_logical_clock(check_plant_events)

    def test_processing_time(self):
        if False:
            for i in range(10):
                print('nop')
        withtimestamps_processing_time.withtimestamps_processing_time(check_plant_processing_times)

    def test_time_tuple2unix_time(self):
        if False:
            while True:
                i = 10
        unix_time = withtimestamps.time_tuple2unix_time()
        self.assertIsInstance(unix_time, float)

    def test_datetime2unix_time(self):
        if False:
            return 10
        unix_time = withtimestamps.datetime2unix_time()
        self.assertIsInstance(unix_time, float)
if __name__ == '__main__':
    unittest.main()