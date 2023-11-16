"""A word-counting workflow."""
import logging
import unittest
import uuid
import pytest
from hamcrest.core.core.allof import all_of
from apache_beam.io.gcp.tests.pubsub_matcher import PubSubMessageMatcher
from apache_beam.runners.dataflow import dataflow_exercise_streaming_metrics_pipeline
from apache_beam.runners.runner import PipelineState
from apache_beam.testing import metric_result_matchers
from apache_beam.testing import test_utils
from apache_beam.testing.metric_result_matchers import DistributionMatcher
from apache_beam.testing.metric_result_matchers import MetricResultMatcher
from apache_beam.testing.pipeline_verifiers import PipelineStateMatcher
from apache_beam.testing.test_pipeline import TestPipeline
INPUT_TOPIC = 'exercise_streaming_metrics_topic_input'
INPUT_SUB = 'exercise_streaming_metrics_subscription_input'
OUTPUT_TOPIC = 'exercise_streaming_metrics_topic_output'
OUTPUT_SUB = 'exercise_streaming_metrics_subscription_output'
WAIT_UNTIL_FINISH_DURATION = 5 * 60 * 1000
MESSAGES_TO_PUBLISH = ['message a', 'message b b', 'message c']
SLEEP_TIME_SECS = 1
_LOGGER = logging.getLogger(__name__)

class ExerciseStreamingMetricsPipelineTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        'Creates all required topics and subs.'
        self.test_pipeline = TestPipeline(is_integration_test=True)
        self.project = self.test_pipeline.get_option('project')
        self.uuid = str(uuid.uuid4())
        from google.cloud import pubsub
        self.pub_client = pubsub.PublisherClient()
        self.input_topic_name = INPUT_TOPIC + self.uuid
        self.input_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, self.input_topic_name))
        self.output_topic_name = OUTPUT_TOPIC + self.uuid
        self.output_topic = self.pub_client.create_topic(name=self.pub_client.topic_path(self.project, self.output_topic_name))
        self.sub_client = pubsub.SubscriberClient()
        self.input_sub_name = INPUT_SUB + self.uuid
        self.input_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, self.input_sub_name), topic=self.input_topic.name)
        self.output_sub_name = OUTPUT_SUB + self.uuid
        self.output_sub = self.sub_client.create_subscription(name=self.sub_client.subscription_path(self.project, self.output_sub_name), topic=self.output_topic.name, ack_deadline_seconds=60)

    def _inject_words(self, topic, messages):
        if False:
            print('Hello World!')
        'Inject messages as test data to PubSub.'
        _LOGGER.debug('Injecting messages to topic %s', topic.name)
        for msg in messages:
            self.pub_client.publish(self.input_topic.name, msg.encode('utf-8'))
        _LOGGER.debug('Done. Injecting messages to topic %s', topic.name)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Delete all created topics and subs.'
        test_utils.cleanup_subscriptions(self.sub_client, [self.input_sub, self.output_sub])
        test_utils.cleanup_topics(self.pub_client, [self.input_topic, self.output_topic])

    def run_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        expected_msg = [msg.encode('utf-8') for msg in MESSAGES_TO_PUBLISH]
        pubsub_msg_verifier = PubSubMessageMatcher(self.project, self.output_sub.name, expected_msg, timeout=600)
        state_verifier = PipelineStateMatcher(PipelineState.RUNNING)
        extra_opts = {'wait_until_finish_duration': WAIT_UNTIL_FINISH_DURATION, 'on_success_matcher': all_of(state_verifier, pubsub_msg_verifier), 'experiment': 'beam_fn_api', 'input_subscription': self.input_sub.name, 'output_topic': self.output_topic.name}
        argv = self.test_pipeline.get_full_options_as_args(**extra_opts)
        return dataflow_exercise_streaming_metrics_pipeline.run(argv)

    @pytest.mark.it_validatesrunner
    @pytest.mark.no_sickbay_batch
    @pytest.mark.no_xdist
    def test_streaming_pipeline_returns_expected_user_metrics_fnapi_it(self):
        if False:
            i = 10
            return i + 15
        '\n    Runs streaming Dataflow job and verifies that user metrics are reported\n    correctly.\n    '
        self._inject_words(self.input_topic, MESSAGES_TO_PUBLISH)
        result = self.run_pipeline()
        METRIC_NAMESPACE = 'apache_beam.runners.dataflow.dataflow_exercise_streaming_metrics_pipeline.StreamingUserMetricsDoFn'
        matchers = [MetricResultMatcher(name='ElementCount', labels={'output_user_name': 'generate_metrics-out0', 'original_name': 'generate_metrics-out0-ElementCount'}, attempted=len(MESSAGES_TO_PUBLISH), committed=len(MESSAGES_TO_PUBLISH)), MetricResultMatcher(name='double_msg_counter_name', namespace=METRIC_NAMESPACE, step='generate_metrics', attempted=len(MESSAGES_TO_PUBLISH) * 2, committed=len(MESSAGES_TO_PUBLISH) * 2), MetricResultMatcher(name='msg_len_dist_metric_name', namespace=METRIC_NAMESPACE, step='generate_metrics', attempted=DistributionMatcher(sum_value=len(''.join(MESSAGES_TO_PUBLISH)), count_value=len(MESSAGES_TO_PUBLISH), min_value=len(MESSAGES_TO_PUBLISH[0]), max_value=len(MESSAGES_TO_PUBLISH[1])), committed=DistributionMatcher(sum_value=len(''.join(MESSAGES_TO_PUBLISH)), count_value=len(MESSAGES_TO_PUBLISH), min_value=len(MESSAGES_TO_PUBLISH[0]), max_value=len(MESSAGES_TO_PUBLISH[1])))]
        metrics = result.metrics().all_metrics()
        errors = metric_result_matchers.verify_all(metrics, matchers)
        self.assertFalse(errors, str(errors))
if __name__ == '__main__':
    unittest.main()