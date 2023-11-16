"""Test for the leader_board example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.complete.game import leader_board
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class LeaderBoardTest(unittest.TestCase):
    SAMPLE_DATA = ['user1_team1,team1,18,1447686663000,2015-11-16 15:11:03.921', 'user1_team1,team1,18,1447690263000,2015-11-16 16:11:03.921', 'user2_team2,team2,2,1447690263000,2015-11-16 16:11:03.955', 'user3_team3,team3,8,1447690263000,2015-11-16 16:11:03.955', 'user4_team3,team3,5,1447690263000,2015-11-16 16:11:03.959', 'user1_team1,team1,14,1447697463000,2015-11-16 18:11:03.955']

    def create_data(self, p):
        if False:
            while True:
                i = 10
        return p | beam.Create(LeaderBoardTest.SAMPLE_DATA) | beam.ParDo(leader_board.ParseGameEventFn()) | beam.Map(lambda elem: beam.window.TimestampedValue(elem, elem['timestamp']))

    def test_leader_board_teams(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline() as p:
            result = self.create_data(p) | leader_board.CalculateTeamScores(team_window_duration=60, allowed_lateness=120)
            assert_that(result, equal_to([('team1', 14), ('team1', 18), ('team1', 18), ('team2', 2), ('team3', 13)]))

    def test_leader_board_users(self):
        if False:
            for i in range(10):
                print('nop')
        test_options = PipelineOptions(flags=['--allow_unsafe_triggers'])
        with TestPipeline(options=test_options) as p:
            result = self.create_data(p) | leader_board.CalculateUserScores(allowed_lateness=120)
            assert_that(result, equal_to([]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()