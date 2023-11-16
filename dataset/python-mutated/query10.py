"""
Query 10, 'Log to sharded files' (Not in original suite.)

Every window_size_sec, save all events from the last period into
2*max_workers log files.
"""
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms import trigger
from apache_beam.transforms import window
from apache_beam.utils.timestamp import Duration
NUM_SHARD_PER_WORKER = 5
LATE_BATCHING_PERIOD = 10
output_path = None
max_num_workers = 5
num_log_shards = NUM_SHARD_PER_WORKER * max_num_workers

class OutputFile(object):

    def __init__(self, max_timestamp, shard, index, timing, filename):
        if False:
            for i in range(10):
                print('nop')
        self.max_timestamp = max_timestamp
        self.shard = shard
        self.index = index
        self.timing = timing
        self.filename = filename

def open_writable_gcs_file(options, filename):
    if False:
        return 10
    pass

def output_file_for(window, shard, pane):
    if False:
        return 10
    '\n  Returns:\n    an OutputFile object constructed with pane, window and shard.\n  '
    filename = '%s/LOG-%s-%s-%03d-%s' % (output_path, window.max_timestamp(), shard, pane.index, pane.timing) if output_path else None
    return OutputFile(window.max_timestamp(), shard, pane.index, pane.timing, filename)

def index_path_for(window):
    if False:
        print('Hello World!')
    '\n  Returns:\n    path to the index file containing all shard names or None if no output_path\n      is set\n  '
    if output_path:
        return '%s/INDEX-%s' % (output_path, window.max_timestamp())
    else:
        return None

def load(events, metadata=None, pipeline_options=None):
    if False:
        return 10
    return events | 'query10_shard_events' >> beam.ParDo(ShardEventsDoFn()) | 'query10_fix_window' >> beam.WindowInto(window.FixedWindows(metadata.get('window_size_sec')), trigger=trigger.AfterEach(trigger.OrFinally(trigger.Repeatedly(trigger.AfterCount(metadata.get('max_log_events'))), trigger.AfterWatermark()), trigger.Repeatedly(trigger.AfterAny(trigger.AfterCount(metadata.get('max_log_events')), trigger.AfterProcessingTime(LATE_BATCHING_PERIOD)))), accumulation_mode=trigger.AccumulationMode.DISCARDING, allowed_lateness=Duration.of(1 * 24 * 60 * 60)) | 'query10_gbk' >> beam.GroupByKey() | 'query10_write_event' >> beam.ParDo(WriteEventDoFn(), pipeline_options) | 'query10_window_log_files' >> beam.WindowInto(window.FixedWindows(metadata.get('window_size_sec')), accumulation_mode=trigger.AccumulationMode.DISCARDING, allowed_lateness=Duration.of(1 * 24 * 60 * 60)) | 'query10_gbk_2' >> beam.GroupByKey() | 'query10_write_index' >> beam.ParDo(WriteIndexDoFn(), pipeline_options)

class ShardEventsDoFn(beam.DoFn):

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        shard_number = abs(hash(element) % num_log_shards)
        shard = 'shard-%05d-of-%05d' % (shard_number, num_log_shards)
        yield (shard, element)

class WriteEventDoFn(beam.DoFn):

    def process(self, element, pipeline_options, window=beam.DoFn.WindowParam, pane_info=beam.DoFn.PaneInfoParam):
        if False:
            print('Hello World!')
        shard = element[0]
        options = pipeline_options.view_as(GoogleCloudOptions)
        output_file = output_file_for(window, shard, pane_info)
        if output_file.filename:
            open_writable_gcs_file(options, output_file.filename)
            for event in element[1]:
                pass
        yield (None, output_file)

class WriteIndexDoFn(beam.DoFn):

    def process(self, element, pipeline_options, window=beam.DoFn.WindowParam):
        if False:
            print('Hello World!')
        options = pipeline_options.view_as(GoogleCloudOptions)
        filename = index_path_for(window)
        if filename:
            open_writable_gcs_file(options, filename)
            for output_file in element[1]:
                pass