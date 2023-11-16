"""An example that reads Wikipedia edit data and computes strings of edits.

An example that reads Wikipedia edit data from Cloud Storage and computes the
user with the longest string of edits separated by no more than an hour within
each 30 day period.

To execute this pipeline locally using the DirectRunner, specify an
output prefix on GCS:::

  --output gs://YOUR_OUTPUT_PREFIX

To execute this pipeline using the Google Cloud Dataflow service, specify
pipeline configuration in addition to the above:::

  --job_name NAME_FOR_YOUR_JOB
  --project YOUR_PROJECT_ID
  --region GCE_REGION
  --staging_location gs://YOUR_STAGING_DIRECTORY
  --temp_location gs://YOUR_TEMPORARY_DIRECTORY
  --runner DataflowRunner

The default input is ``gs://dataflow-samples/wikipedia_edits/*.json`` and can
be overridden with --input.
"""
import argparse
import json
import logging
import apache_beam as beam
from apache_beam import combiners
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.transforms.window import FixedWindows
from apache_beam.transforms.window import Sessions
from apache_beam.transforms.window import TimestampedValue
ONE_HOUR_IN_SECONDS = 3600
THIRTY_DAYS_IN_SECONDS = 30 * 24 * ONE_HOUR_IN_SECONDS
MAX_TIMESTAMP = 9223372036854775807

class ExtractUserAndTimestampDoFn(beam.DoFn):
    """Extracts user and timestamp representing a Wikipedia edit."""

    def process(self, element):
        if False:
            while True:
                i = 10
        table_row = json.loads(element)
        if 'contributor_username' in table_row:
            user_name = table_row['contributor_username']
            timestamp = table_row['timestamp']
            yield TimestampedValue(user_name, timestamp)

class ComputeSessions(beam.PTransform):
    """Computes the number of edits in each user session.

  A session is defined as a string of edits where each is separated from the
  next by less than an hour.
  """

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | 'ComputeSessionsWindow' >> beam.WindowInto(Sessions(gap_size=ONE_HOUR_IN_SECONDS)) | combiners.Count.PerElement()

class TopPerMonth(beam.PTransform):
    """Computes the longest session ending in each month."""

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | 'TopPerMonthWindow' >> beam.WindowInto(FixedWindows(size=THIRTY_DAYS_IN_SECONDS)) | 'Top' >> combiners.core.CombineGlobally(combiners.TopCombineFn(n=10, key=lambda sessions_count: sessions_count[1])).without_defaults()

class SessionsToStringsDoFn(beam.DoFn):
    """Adds the session information to be part of the key."""

    def process(self, element, window=beam.DoFn.WindowParam):
        if False:
            while True:
                i = 10
        yield (element[0] + ' : ' + str(window), element[1])

class FormatOutputDoFn(beam.DoFn):
    """Formats a string containing the user, count, and session."""

    def process(self, element, window=beam.DoFn.WindowParam):
        if False:
            for i in range(10):
                print('nop')
        for kv in element:
            session = kv[0]
            count = kv[1]
            yield (session + ' : ' + str(count) + ' : ' + str(window))

class ComputeTopSessions(beam.PTransform):
    """Computes the top user sessions for each month."""

    def __init__(self, sampling_threshold):
        if False:
            i = 10
            return i + 15
        beam.PTransform.__init__(self)
        self.sampling_threshold = sampling_threshold

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | 'ExtractUserAndTimestamp' >> beam.ParDo(ExtractUserAndTimestampDoFn()) | beam.Filter(lambda x: abs(hash(x)) <= MAX_TIMESTAMP * self.sampling_threshold) | ComputeSessions() | 'SessionsToStrings' >> beam.ParDo(SessionsToStringsDoFn()) | TopPerMonth() | 'FormatOutput' >> beam.ParDo(FormatOutputDoFn())

def run(argv=None):
    if False:
        return 10
    'Runs the Wikipedia top edits pipeline.\n\n  Args:\n    argv: Pipeline options as a list of arguments.\n  '
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/wikipedia_edits/*.json', help='Input specified as a GCS path containing a BigQuery table exported as json.')
    parser.add_argument('--output', required=True, help='Output file to write results to.')
    parser.add_argument('--sampling_threshold', type=float, default=0.1, help='Fraction of entries used for session tracking')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    with beam.Pipeline(options=pipeline_options) as p:
        p | ReadFromText(known_args.input) | ComputeTopSessions(known_args.sampling_threshold) | WriteToText(known_args.output)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()