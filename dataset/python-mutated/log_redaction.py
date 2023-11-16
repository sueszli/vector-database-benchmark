from __future__ import annotations
import argparse
import json
import logging
from apache_beam import CombineFn, CombineGlobally, DoFn, io, ParDo, Pipeline, WindowInto
from apache_beam.error import PipelineError
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.window import FixedWindows
from google.cloud import logging_v2

class PayloadAsJson(DoFn):
    """Convert PubSub message payload to UTF-8 and return as JSON"""

    def process(self, element):
        if False:
            while True:
                i = 10
        yield json.loads(element.decode('utf-8'))

class BatchPayloads(CombineFn):
    """Opinionated way to batch all payloads in the window"""

    def create_accumulator(self):
        if False:
            while True:
                i = 10
        return []

    def add_input(self, accumulator, input):
        if False:
            while True:
                i = 10
        accumulator.append(input)
        return accumulator

    def merge_accumulators(self, accumulators):
        if False:
            while True:
                i = 10
        merged = [item for accumulator in accumulators for item in accumulator]
        return merged

    def extract_output(self, accumulator):
        if False:
            return 10
        return accumulator

class IngestLogs(DoFn):
    """Ingest payloads into destination log"""

    def __init__(self, destination_log_name):
        if False:
            return 10
        self.destination_log_name = destination_log_name
        self.logger = None

    def _replace_log_name(self, entry):
        if False:
            i = 10
            return i + 15
        entry['logName'] = self.logger.name
        return entry

    def setup(self):
        if False:
            return 10
        if self.logger:
            return
        logging_client = logging_v2.Client()
        if not logging_client:
            logging.error('Cannot create GCP Logging Client')
            raise PipelineError('Cannot create GCP Logging Client')
        self.logger = logging_client.logger(self.destination_log_name)
        if not self.logger:
            logging.error('Google client library cannot create Logger object')
            raise PipelineError('Google client library cannot create Logger object')

    def process(self, element):
        if False:
            print('Hello World!')
        if self.logger:
            logs = list(map(self._replace_log_name, element))
            self.logger.client.logging_api.write_entries(logs)
        yield logs

def run(pubsub_subscription: str, destination_log_name: str, window_size: float, pipeline_args: list[str]=None) -> None:
    if False:
        while True:
            i = 10
    'Runs Dataflow pipeline'
    pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
    pipeline = Pipeline(options=pipeline_options)
    _ = pipeline | 'Read log entries from Pub/Sub' >> io.ReadFromPubSub(subscription=pubsub_subscription) | 'Convert log entry payload to Json' >> ParDo(PayloadAsJson()) | 'Aggregate payloads in fixed time intervals' >> WindowInto(FixedWindows(window_size)) | 'Batch aggregated payloads' >> CombineGlobally(BatchPayloads()).without_defaults() | 'Ingest to output log' >> ParDo(IngestLogs(destination_log_name))
    pipeline.run()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubsub_subscription', help='The Cloud Pub/Sub subscription to read from in the format "projects/<PROJECT_ID>/subscription/<SUBSCRIPTION_ID>".')
    parser.add_argument('--destination_log_name', help='The log name to ingest log entries in the format "projects/<PROJECT_ID>/logs/<LOG_ID>".')
    parser.add_argument('--window_size', type=float, default=60.0, help="Output file's window size in seconds.")
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.pubsub_subscription, known_args.destination_log_name, known_args.window_size, pipeline_args)