"""End-to-end test for the streaming wordcount example with debug."""
import logging
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.examples import streaming_wordcount_debugging
from apache_beam.io.gcp.tests.pubsub_matcher import PubSubMessageMatcher
from apache_beam.runners.runner import PipelineState
from apache_beam.testing import test_utils
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
INPUT_TOPIC = 'wc_topic_input'
OUTPUT_TOPIC = 'wc_topic_output'
INPUT_SUB = 'wc_subscription_input'
OUTPUT_SUB = 'wc_subscription_output'
SAMPLE_MESSAGES = ['150', '151', '152', '153', '154', '210', '211', '212', '213', '214']
EXPECTED_MESSAGE = ['150: 1', '151: 1', '152: 1', '153: 1', '154: 1', '210: 1', '211: 1', '212: 1', '213: 1', '214: 1']
WAIT_UNTIL_FINISH_DURATION = 6 * 60 * 1000

class StreamingWordcountDebuggingIT(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.project = self.test_pipeline.get_option('project')
        self.setup_pubsub()

    def setup_pubsub(self):
        if False:
            while True:
                i = 10
        self.uuid = str(uuid.uuid4())
        from google.cloud import pubsub
        self.pub_client = pubsub.PublisherClient()
        self.input_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, INPUT_TOPIC + self.uuid))
        self.output_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, OUTPUT_TOPIC + self.uuid))
        self.sub_client = pubsub.SubscriberClient()
        self.input_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, INPUT_SUB + self.uuid), topic=self.input_topic.name)
        self.output_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, OUTPUT_SUB + self.uuid), topic=self.output_topic.name, ack_deadline_seconds=60)

    def _inject_data(self, topic, data):
        if False:
            print('Hello World!')
        'Inject numbers as test data to PubSub.'
        logging.debug('Injecting test data to topic %s', topic.name)
        for n in data:
            self.pub_client.publish(self.input_topic.name, str(n).encode('utf-8'))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        test_utils.cleanup_subscriptions(self.sub_client, [self.input_sub, self.output_sub])
        test_utils.cleanup_topics(self.pub_client, [self.input_topic, self.output_topic])

    @pytest.mark.it_postcommit
    @unittest.skip('Skipped due to [https://github.com/apache/beam/issues/18709]: assert_that not working for streaming')
    def test_streaming_wordcount_debugging_it(self):
        if False:
            for i in range(10):
                print('nop')
        state_verifier = PipelineStateMatcher(PipelineState.RUNNING)
        pubsub_msg_verifier = PubSubMessageMatcher(self.project, self.output_sub.name, EXPECTED_MESSAGE, timeout=400)
        extra_opts = {'input_subscription': self.input_sub.name, 'output_topic': self.output_topic.name, 'wait_until_finish_duration': WAIT_UNTIL_FINISH_DURATION, 'on_success_matcher': all_of(state_verifier, pubsub_msg_verifier)}
        self._inject_data(self.input_topic, SAMPLE_MESSAGES)
        streaming_wordcount_debugging.run(self.test_pipeline.get_full_options_as_args(**extra_opts), save_main_session=False)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()