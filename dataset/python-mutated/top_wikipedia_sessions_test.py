"""Test for the top wikipedia sessions example."""
import json
import unittest
import apache_beam as beam
from apache_beam.examples.complete import top_wikipedia_sessions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class ComputeTopSessionsTest(unittest.TestCase):
    EDITS = [json.dumps({'timestamp': 0.0, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.001, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.002, 'contributor_username': 'user1'}), json.dumps({'timestamp': 0.0, 'contributor_username': 'user2'}), json.dumps({'timestamp': 0.001, 'contributor_username': 'user2'}), json.dumps({'timestamp': 3.601, 'contributor_username': 'user2'}), json.dumps({'timestamp': 3.602, 'contributor_username': 'user2'}), json.dumps({'timestamp': 2 * 3600.0, 'contributor_username': 'user2'}), json.dumps({'timestamp': 35 * 24 * 3.6, 'contributor_username': 'user3'})]
    EXPECTED = ['user1 : [0.0, 3600.002) : 3 : [0.0, 2592000.0)', 'user2 : [0.0, 3603.602) : 4 : [0.0, 2592000.0)', 'user2 : [7200.0, 10800.0) : 1 : [0.0, 2592000.0)', 'user3 : [3024.0, 6624.0) : 1 : [0.0, 2592000.0)']

    def test_compute_top_sessions(self):
        if False:
            print('Hello World!')
        with TestPipeline() as p:
            edits = p | beam.Create(self.EDITS)
            result = edits | top_wikipedia_sessions.ComputeTopSessions(1.0)
            assert_that(result, equal_to(self.EXPECTED))
if __name__ == '__main__':
    unittest.main()