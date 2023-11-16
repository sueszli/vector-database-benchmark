"""End-to-end test for the streaming wordcount example."""
import logging
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples import streaming_wordcount
from apache_beam.io.gcp.tests.pubsub_matcher import PubSubMessageMatcher
from apache_beam.runners.runner import PipelineState
from apache_beam.testing import test_utils
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
INPUT_TOPIC = 'wc_topic_input'
OUTPUT_TOPIC = 'wc_topic_output'
INPUT_SUB = 'wc_subscription_input'
OUTPUT_SUB = 'wc_subscription_output'
DEFAULT_INPUT_NUMBERS = 500
WAIT_UNTIL_FINISH_DURATION = 10 * 60 * 1000

class StreamingWordCountIT(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.project = self.test_pipeline.get_option('project')
        self.uuid = str(uuid.uuid4())
        from google.cloud import pubsub
        self.pub_client = pubsub.PublisherClient()
        self.input_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, INPUT_TOPIC + self.uuid))
        self.output_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, OUTPUT_TOPIC + self.uuid))
        self.sub_client = pubsub.SubscriberClient()
        self.input_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, INPUT_SUB + self.uuid), topic=self.input_topic.name)
        self.output_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, OUTPUT_SUB + self.uuid), topic=self.output_topic.name, ack_deadline_seconds=60)

    def _inject_numbers(self, topic, num_messages):
        if False:
            while True:
                i = 10
        'Inject numbers as test data to PubSub.'
        logging.debug('Injecting %d numbers to topic %s', num_messages, topic.name)
        for n in range(num_messages):
            self.pub_client.publish(self.input_topic.name, str(n).encode('utf-8'))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        test_utils.cleanup_subscriptions(self.sub_client, [self.input_sub, self.output_sub])
        test_utils.cleanup_topics(self.pub_client, [self.input_topic, self.output_topic])

    @pytest.mark.it_postcommit
    def test_streaming_wordcount_it(self):
        if False:
            i = 10
            return i + 15
        expected_msg = [('%d: 1' % num).encode('utf-8') for num in range(DEFAULT_INPUT_NUMBERS)]
        state_verifier = PipelineStateMatcher(PipelineState.RUNNING)
        pubsub_msg_verifier = PubSubMessageMatcher(self.project, self.output_sub.name, expected_msg, timeout=400)
        extra_opts = {'input_subscription': self.input_sub.name, 'output_topic': self.output_topic.name, 'wait_until_finish_duration': WAIT_UNTIL_FINISH_DURATION, 'on_success_matcher': all_of(state_verifier, pubsub_msg_verifier)}
        self._inject_numbers(self.input_topic, DEFAULT_INPUT_NUMBERS)
        streaming_wordcount.run(self.test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()