"""End-to-end test for the game stats example.

Code: beam/sdks/python/apache_beam/examples/complete/game/game_stats.py
Usage:

  pytest --test-pipeline-options="       --runner=TestDataflowRunner       --project=...       --region=...       --staging_location=gs://...       --temp_location=gs://...       --output=gs://...       --sdk_location=... 
"""
import logging
import time
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples.complete.game import game_stats
from apache_beam.io.gcp.tests import utils
from apache_beam.io.gcp.tests.bigquery_matcher import BigqueryMatcher
from apache_beam.runners.runner import PipelineState
from apache_beam.testing import test_utils
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline

class GameStatsIT(unittest.TestCase):
    INPUT_EVENT = 'user1,teamA,10,%d,2015-11-02 09:09:28.224'
    INPUT_TOPIC = 'game_stats_it_input_topic'
    INPUT_SUB = 'game_stats_it_input_subscription'
    DEFAULT_EXPECTED_CHECKSUM = '5288ccaab77d347c8460d77c15a0db234ef5eb4f'
    OUTPUT_DATASET = 'game_stats_it_dataset'
    OUTPUT_TABLE_SESSIONS = 'game_stats_sessions'
    OUTPUT_TABLE_TEAMS = 'game_stats_teams'
    DEFAULT_INPUT_COUNT = 500
    WAIT_UNTIL_FINISH_DURATION = 12 * 60 * 1000

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.project = self.test_pipeline.get_option('project')
        _unique_id = str(uuid.uuid4())
        from google.cloud import pubsub
        self.pub_client = pubsub.PublisherClient()
        self.input_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, self.INPUT_TOPIC + _unique_id))
        self.sub_client = pubsub.SubscriberClient()
        self.input_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, self.INPUT_SUB + _unique_id), topic=self.input_topic.name)
        self.dataset_ref = utils.create_bq_dataset(self.project, self.OUTPUT_DATASET)
        self._test_timestamp = int(time.time() * 1000)

    def _inject_pubsub_game_events(self, topic, message_count):
        if False:
            while True:
                i = 10
        'Inject game events as test data to PubSub.'
        logging.debug('Injecting %d game events to topic %s', message_count, topic.name)
        for _ in range(message_count):
            self.pub_client.publish(topic.name, (self.INPUT_EVENT % self._test_timestamp).encode('utf-8'))

    def _cleanup_pubsub(self):
        if False:
            while True:
                i = 10
        test_utils.cleanup_subscriptions(self.sub_client, [self.input_sub])
        test_utils.cleanup_topics(self.pub_client, [self.input_topic])

    @pytest.mark.it_postcommit
    @pytest.mark.examples_postcommit
    @pytest.mark.sickbay_direct
    @pytest.mark.sickbay_spark
    @pytest.mark.sickbay_flink
    def test_game_stats_it(self):
        if False:
            i = 10
            return i + 15
        state_verifier = PipelineStateMatcher(PipelineState.RUNNING)
        success_condition = 'mean_duration=300 LIMIT 1'
        sessions_query = 'SELECT mean_duration FROM `%s.%s.%s` WHERE %s' % (self.project, self.dataset_ref.dataset_id, self.OUTPUT_TABLE_SESSIONS, success_condition)
        bq_sessions_verifier = BigqueryMatcher(self.project, sessions_query, self.DEFAULT_EXPECTED_CHECKSUM)
        extra_opts = {'subscription': self.input_sub.name, 'dataset': self.dataset_ref.dataset_id, 'topic': self.input_topic.name, 'fixed_window_duration': 1, 'user_activity_window_duration': 1, 'wait_until_finish_duration': self.WAIT_UNTIL_FINISH_DURATION, 'on_success_matcher': all_of(state_verifier, bq_sessions_verifier)}
        self.addCleanup(self._cleanup_pubsub)
        self.addCleanup(utils.delete_bq_dataset, self.project, self.dataset_ref)
        self._inject_pubsub_game_events(self.input_topic, self.DEFAULT_INPUT_COUNT)
        game_stats.run(self.test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()