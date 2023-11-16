"""Test for the game_stats example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.complete.game import game_stats
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class GameStatsTest(unittest.TestCase):
    SAMPLE_DATA = ['user1_team1,team1,18,1447686663000,2015-11-16 15:11:03.921', 'user1_team1,team1,18,1447690263000,2015-11-16 16:11:03.921', 'user2_team2,team2,2,1447690263000,2015-11-16 16:11:03.955', 'user3_team3,team3,8,1447690263000,2015-11-16 16:11:03.955', 'user4_team3,team3,5,1447690263000,2015-11-16 16:11:03.959', 'user1_team1,team1,14,1447697463000,2015-11-16 18:11:03.955', 'robot1_team1,team1,9000,1447697463000,2015-11-16 18:11:03.955', 'robot2_team2,team2,1,1447697463000,2015-11-16 20:11:03.955', 'robot2_team2,team2,9000,1447697463000,2015-11-16 21:11:03.955']

    def create_data(self, p):
        if False:
            while True:
                i = 10
        return p | beam.Create(GameStatsTest.SAMPLE_DATA) | beam.ParDo(game_stats.ParseGameEventFn()) | beam.Map(lambda elem: beam.window.TimestampedValue(elem, elem['timestamp']))

    def test_spammy_users(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline() as p:
            result = self.create_data(p) | beam.Map(lambda elem: (elem['user'], elem['score'])) | game_stats.CalculateSpammyUsers()
            assert_that(result, equal_to([('robot1_team1', 9000), ('robot2_team2', 9001)]))

    def test_game_stats_sessions(self):
        if False:
            return 10
        session_gap = 5 * 60
        user_activity_window_duration = 30 * 60
        with TestPipeline() as p:
            result = self.create_data(p) | beam.Map(lambda elem: (elem['user'], elem['score'])) | 'WindowIntoSessions' >> beam.WindowInto(beam.window.Sessions(session_gap), timestamp_combiner=beam.window.TimestampCombiner.OUTPUT_AT_EOW) | beam.CombinePerKey(lambda _: None) | beam.ParDo(game_stats.UserSessionActivity()) | 'WindowToExtractSessionMean' >> beam.WindowInto(beam.window.FixedWindows(user_activity_window_duration)) | beam.CombineGlobally(beam.combiners.MeanCombineFn()).without_defaults()
            assert_that(result, equal_to([300.0, 300.0, 300.0]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()