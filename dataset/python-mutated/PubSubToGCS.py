import argparse
from datetime import datetime
import logging
import random
from apache_beam import DoFn, GroupByKey, io, ParDo, Pipeline, PTransform, WindowInto, WithKeys
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.window import FixedWindows

class GroupMessagesByFixedWindows(PTransform):
    """A composite transform that groups Pub/Sub messages based on publish time
    and outputs a list of tuples, each containing a message and its publish time.
    """

    def __init__(self, window_size, num_shards=5):
        if False:
            for i in range(10):
                print('nop')
        self.window_size = int(window_size * 60)
        self.num_shards = num_shards

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | 'Window into fixed intervals' >> WindowInto(FixedWindows(self.window_size)) | 'Add timestamp to windowed elements' >> ParDo(AddTimestamp()) | 'Add key' >> WithKeys(lambda _: random.randint(0, self.num_shards - 1)) | 'Group by key' >> GroupByKey()

class AddTimestamp(DoFn):

    def process(self, element, publish_time=DoFn.TimestampParam):
        if False:
            i = 10
            return i + 15
        'Processes each windowed element by extracting the message body and its\n        publish time into a tuple.\n        '
        yield (element.decode('utf-8'), datetime.utcfromtimestamp(float(publish_time)).strftime('%Y-%m-%d %H:%M:%S.%f'))

class WriteToGCS(DoFn):

    def __init__(self, output_path):
        if False:
            for i in range(10):
                print('nop')
        self.output_path = output_path

    def process(self, key_value, window=DoFn.WindowParam):
        if False:
            return 10
        'Write messages in a batch to Google Cloud Storage.'
        ts_format = '%H:%M'
        window_start = window.start.to_utc_datetime().strftime(ts_format)
        window_end = window.end.to_utc_datetime().strftime(ts_format)
        (shard_id, batch) = key_value
        filename = '-'.join([self.output_path, window_start, window_end, str(shard_id)])
        with io.gcsio.GcsIO().open(filename=filename, mode='w') as f:
            for (message_body, publish_time) in batch:
                f.write(f'{message_body},{publish_time}\n'.encode())

def run(input_topic, output_path, window_size=1.0, num_shards=5, pipeline_args=None):
    if False:
        i = 10
        return i + 15
    pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
    with Pipeline(options=pipeline_options) as pipeline:
        pipeline | 'Read from Pub/Sub' >> io.ReadFromPubSub(topic=input_topic) | 'Window into' >> GroupMessagesByFixedWindows(window_size, num_shards) | 'Write to GCS' >> ParDo(WriteToGCS(output_path))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_topic', help='The Cloud Pub/Sub topic to read from."projects/<PROJECT_ID>/topics/<TOPIC_ID>".')
    parser.add_argument('--window_size', type=float, default=1.0, help="Output file's window size in minutes.")
    parser.add_argument('--output_path', help='Path of the output GCS file including the prefix.')
    parser.add_argument('--num_shards', type=int, default=5, help='Number of shards to use when writing windowed elements to GCS.')
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.input_topic, known_args.output_path, known_args.window_size, known_args.num_shards, pipeline_args)