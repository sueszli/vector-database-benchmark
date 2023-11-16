"""A word-counting workflow."""
import argparse
import logging
import time
import apache_beam as beam
from apache_beam.metrics import Metrics
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
SLEEP_TIME_SECS = 1
_LOGGER = logging.getLogger(__name__)

class StreamingUserMetricsDoFn(beam.DoFn):
    """Generates user metrics and outputs same element."""

    def __init__(self):
        if False:
            return 10
        self.double_message_counter = Metrics.counter(self.__class__, 'double_msg_counter_name')
        self.msg_len_dist_metric = Metrics.distribution(self.__class__, 'msg_len_dist_metric_name')

    def start_bundle(self):
        if False:
            return 10
        time.sleep(SLEEP_TIME_SECS)

    def process(self, element):
        if False:
            while True:
                i = 10
        'Returns the processed element and increments the metrics.'
        text_line = element.strip()
        self.double_message_counter.inc()
        self.double_message_counter.inc()
        self.msg_len_dist_metric.update(len(text_line))
        _LOGGER.debug("Done processing returning element array: '%s'", element)
        return [element]

    def finish_bundle(self):
        if False:
            return 10
        time.sleep(SLEEP_TIME_SECS)

def run(argv=None):
    if False:
        return 10
    'Given an initialized Pipeline applies transforms and runs it.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_topic', required=True, help='Output PubSub topic of the form "projects/<PROJECT>/topic/<TOPIC>".')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_topic', help='Input PubSub topic of the form "projects/<PROJECT>/topics/<TOPIC>".')
    group.add_argument('--input_subscription', help='Input PubSub subscription of the form "projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True
    pipeline = beam.Pipeline(options=pipeline_options)
    _ = pipeline | beam.io.ReadFromPubSub(subscription=known_args.input_subscription) | 'generate_metrics' >> beam.ParDo(StreamingUserMetricsDoFn()) | 'dump_to_pub' >> beam.io.WriteToPubSub(known_args.output_topic)
    result = pipeline.run()
    result.wait_until_finish()
    return result
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()