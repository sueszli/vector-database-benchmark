"""Test for the user_score example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.complete.game import user_score
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class UserScoreTest(unittest.TestCase):
    SAMPLE_DATA = ['user1_team1,team1,18,1447686663000,2015-11-16 15:11:03.921', 'user1_team1,team1,18,1447690263000,2015-11-16 16:11:03.921', 'user2_team2,team2,2,1447690263000,2015-11-16 16:11:03.955', 'user3_team3,team3,8,1447690263000,2015-11-16 16:11:03.955', 'user4_team3,team3,5,1447690263000,2015-11-16 16:11:03.959', 'user1_team1,team1,14,1447697463000,2015-11-16 18:11:03.955']

    def test_user_score(self):
        if False:
            while True:
                i = 10
        with TestPipeline() as p:
            result = p | beam.Create(UserScoreTest.SAMPLE_DATA) | user_score.UserScore()
            assert_that(result, equal_to([('user1_team1', 50), ('user2_team2', 2), ('user3_team3', 8), ('user4_team3', 5)]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()