from __future__ import annotations
import argparse
import json
import logging
from apache_beam import CombineFn, CombineGlobally, DoFn, io, ParDo, Pipeline, WindowInto
from apache_beam.error import PipelineError
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions
from apache_beam.transforms.window import FixedWindows
from google.cloud import dlp_v2, logging_v2
INSPECT_CFG = {'info_types': [{'name': 'US_SOCIAL_SECURITY_NUMBER'}]}
REDACTION_CFG = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'character_mask_config': {'masking_character': '#'}}}]}}

class PayloadAsJson(DoFn):
    """Convert PubSub message payload to UTF-8 and return as JSON"""

    def process(self, element):
        if False:
            return 10
        yield json.loads(element.decode('utf-8'))

class BatchPayloads(CombineFn):
    """Collect all items in the windowed collection into single batch"""

    def create_accumulator(self):
        if False:
            return 10
        return []

    def add_input(self, accumulator, input):
        if False:
            while True:
                i = 10
        accumulator.append(input)
        return accumulator

    def merge_accumulators(self, accumulators):
        if False:
            return 10
        merged = [item for accumulator in accumulators for item in accumulator]
        return merged

    def extract_output(self, accumulator):
        if False:
            while True:
                i = 10
        return accumulator

class LogRedaction(DoFn):
    """Apply inspection and redaction to textPayload field of log entries"""

    def __init__(self, region, project_id: str):
        if False:
            return 10
        self.project_id = project_id
        self.region = region
        self.dlp_client = None

    def _log_to_row(self, entry):
        if False:
            while True:
                i = 10
        payload = entry.get('textPayload', '')
        return {'values': [{'string_value': payload}]}

    def setup(self):
        if False:
            i = 10
            return i + 15
        'Initialize DLP client'
        if self.dlp_client:
            return
        self.dlp_client = dlp_v2.DlpServiceClient()
        if not self.dlp_client:
            logging.error('Cannot create Google DLP Client')
            raise PipelineError('Cannot create Google DLP Client')

    def process(self, logs):
        if False:
            i = 10
            return i + 15
        table = {'table': {'headers': [{'name': 'textPayload'}], 'rows': map(self._log_to_row, logs)}}
        response = self.dlp_client.deidentify_content(request={'parent': f'projects/{self.project_id}/locations/{self.region}', 'inspect_config': INSPECT_CFG, 'deidentify_config': REDACTION_CFG, 'item': table})
        modified_logs = []
        for (index, log) in enumerate(logs):
            log['textPayload'] = response.item.table.rows[index].values[0].string_value
            modified_logs.append(log)
        yield modified_logs

class IngestLogs(DoFn):
    """Ingest payloads into destination log"""

    def __init__(self, destination_log_name):
        if False:
            return 10
        self.destination_log_name = destination_log_name
        self.logger = None

    def _replace_log_name(self, entry):
        if False:
            return 10
        entry['logName'] = self.logger.name
        return entry

    def setup(self):
        if False:
            while True:
                i = 10
        if self.logger:
            return
        logging_client = logging_v2.Client()
        if not logging_client:
            logging.error('Cannot create Google Logging Client')
            raise PipelineError('Cannot create Google Logging Client')
        self.logger = logging_client.logger(self.destination_log_name)
        if not self.logger:
            logging.error('Google client library cannot create Logger object')
            raise PipelineError('Google client library cannot create Logger object')

    def process(self, element):
        if False:
            i = 10
            return i + 15
        if self.logger:
            logs = list(map(self._replace_log_name, element))
            self.logger.client.logging_api.write_entries(logs)
        yield logs

def run(pubsub_subscription: str, destination_log_name: str, window_size: float, pipeline_args: list[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Runs Dataflow pipeline'
    pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
    region = 'us-central1'
    try:
        region = pipeline_options.view_as(GoogleCloudOptions).region
    except AttributeError:
        pass
    pipeline = Pipeline(options=pipeline_options)
    _ = pipeline | 'Read log entries from Pub/Sub' >> io.ReadFromPubSub(subscription=pubsub_subscription) | 'Convert log entry payload to Json' >> ParDo(PayloadAsJson()) | 'Aggregate payloads in fixed time intervals' >> WindowInto(FixedWindows(window_size)) | 'Batch aggregated payloads' >> CombineGlobally(BatchPayloads()).without_defaults() | 'Redact SSN info from logs' >> ParDo(LogRedaction(region, destination_log_name.split('/')[1])) | 'Ingest to output log' >> ParDo(IngestLogs(destination_log_name))
    pipeline.run()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubsub_subscription', help='The Cloud Pub/Sub subscription to read from in the format "projects/<PROJECT_ID>/subscription/<SUBSCRIPTION_ID>".')
    parser.add_argument('--destination_log_name', help='The log name to ingest log entries in the format "projects/<PROJECT_ID>/logs/<LOG_ID>".')
    parser.add_argument('--window_size', type=float, default=60.0, help="Output file's window size in seconds.")
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.pubsub_subscription, known_args.destination_log_name, known_args.window_size, pipeline_args)