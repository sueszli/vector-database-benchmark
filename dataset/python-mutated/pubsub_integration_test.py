"""
Integration test for Google Cloud Pub/Sub.
"""
import logging
import time
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.io.gcp import pubsub_it_pipeline
from apache_beam.io.gcp.pubsub import PubsubMessage
from apache_beam.io.gcp.tests.pubsub_matcher import PubSubMessageMatcher
from apache_beam.runners.runner import PipelineState
from apache_beam.testing import test_utils
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
INPUT_TOPIC = 'psit_topic_input'
OUTPUT_TOPIC = 'psit_topic_output'
INPUT_SUB = 'psit_subscription_input'
OUTPUT_SUB = 'psit_subscription_output'
TEST_PIPELINE_DURATION_MS = 8 * 60 * 1000
MESSAGE_MATCHER_TIMEOUT_S = 5 * 60

class PubSubIntegrationTest(unittest.TestCase):
    ID_LABEL = 'id'
    TIMESTAMP_ATTRIBUTE = 'timestamp'
    INPUT_MESSAGES = {'TestDirectRunner': [PubsubMessage(b'data001', {}), PubsubMessage(b'data002', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z'}), PubsubMessage(b'data003\xab\xac', {}), PubsubMessage(b'data004\xab\xac', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z'})], 'TestDataflowRunner': [PubsubMessage(b'data001', {ID_LABEL: 'foo'}), PubsubMessage(b'data001', {ID_LABEL: 'foo'}), PubsubMessage(b'data001', {ID_LABEL: 'foo'}), PubsubMessage(b'data002', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z'}), PubsubMessage(b'data003\xab\xac', {ID_LABEL: 'foo2'}), PubsubMessage(b'data003\xab\xac', {ID_LABEL: 'foo2'}), PubsubMessage(b'data003\xab\xac', {ID_LABEL: 'foo2'}), PubsubMessage(b'data004\xab\xac', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z'})]}
    EXPECTED_OUTPUT_MESSAGES = {'TestDirectRunner': [PubsubMessage(b'data001-seen', {'processed': 'IT'}), PubsubMessage(b'data002-seen', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z', TIMESTAMP_ATTRIBUTE + '_out': '2018-07-11T02:02:50.149000Z', 'processed': 'IT'}), PubsubMessage(b'data003\xab\xac-seen', {'processed': 'IT'}), PubsubMessage(b'data004\xab\xac-seen', {TIMESTAMP_ATTRIBUTE: '2018-07-11T02:02:50.149000Z', TIMESTAMP_ATTRIBUTE + '_out': '2018-07-11T02:02:50.149000Z', 'processed': 'IT'})], 'TestDataflowRunner': [PubsubMessage(b'data001-seen', {'processed': 'IT'}), PubsubMessage(b'data002-seen', {TIMESTAMP_ATTRIBUTE + '_out': '2018-07-11T02:02:50.149000Z', 'processed': 'IT'}), PubsubMessage(b'data003\xab\xac-seen', {'processed': 'IT'}), PubsubMessage(b'data004\xab\xac-seen', {TIMESTAMP_ATTRIBUTE + '_out': '2018-07-11T02:02:50.149000Z', 'processed': 'IT'})]}

    def setUp(self):
        if False:
            print('Hello World!')
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.runner_name = type(self.test_pipeline.runner).__name__
        self.project = self.test_pipeline.get_option('project')
        self.uuid = str(uuid.uuid4())
        from google.cloud import pubsub
        self.pub_client = pubsub.PublisherClient()
        self.input_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, INPUT_TOPIC + self.uuid))
        self.output_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, OUTPUT_TOPIC + self.uuid))
        self.sub_client = pubsub.SubscriberClient()
        self.input_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, INPUT_SUB + self.uuid), topic=self.input_topic.name)
        self.output_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, OUTPUT_SUB + self.uuid), topic=self.output_topic.name)
        time.sleep(30)

    def tearDown(self):
        if False:
            print('Hello World!')
        test_utils.cleanup_subscriptions(self.sub_client, [self.input_sub, self.output_sub])
        test_utils.cleanup_topics(self.pub_client, [self.input_topic, self.output_topic])

    def _test_streaming(self, with_attributes):
        if False:
            while True:
                i = 10
        'Runs IT pipeline with message verifier.\n\n    Args:\n      with_attributes: False - Reads and writes message data only.\n        True - Reads and writes message data and attributes. Also verifies\n        id_label and timestamp_attribute features.\n    '
        state_verifier = PipelineStateMatcher(PipelineState.RUNNING)
        expected_messages = self.EXPECTED_OUTPUT_MESSAGES[self.runner_name]
        if not with_attributes:
            expected_messages = [pubsub_msg.data for pubsub_msg in expected_messages]
        if self.runner_name == 'TestDirectRunner':
            strip_attributes = None
        else:
            strip_attributes = [self.ID_LABEL, self.TIMESTAMP_ATTRIBUTE]
        pubsub_msg_verifier = PubSubMessageMatcher(self.project, self.output_sub.name, expected_messages, timeout=MESSAGE_MATCHER_TIMEOUT_S, with_attributes=with_attributes, strip_attributes=strip_attributes)
        extra_opts = {'input_subscription': self.input_sub.name, 'output_topic': self.output_topic.name, 'wait_until_finish_duration': TEST_PIPELINE_DURATION_MS, 'on_success_matcher': all_of(state_verifier, pubsub_msg_verifier)}
        for msg in self.INPUT_MESSAGES[self.runner_name]:
            self.pub_client.publish(self.input_topic.name, msg.data, **msg.attributes).result()
        pubsub_it_pipeline.run_pipeline(argv=self.test_pipeline.get_full_options_as_args(**extra_opts), with_attributes=with_attributes, id_label=self.ID_LABEL, timestamp_attribute=self.TIMESTAMP_ATTRIBUTE)

    @pytest.mark.it_postcommit
    def test_streaming_data_only(self):
        if False:
            i = 10
            return i + 15
        self._test_streaming(with_attributes=False)

    @pytest.mark.it_postcommit
    def test_streaming_with_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_streaming(with_attributes=True)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()