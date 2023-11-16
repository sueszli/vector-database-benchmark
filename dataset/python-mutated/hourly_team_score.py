"""Second in a series of four pipelines that tell a story in a 'gaming' domain.

In addition to the concepts introduced in `user_score`, new concepts include:
windowing and element timestamps; use of `Filter`; using standalone DoFns.

This pipeline processes data collected from gaming events in batch, building on
`user_score` but using fixed windows. It calculates the sum of scores per team,
for each window, optionally allowing specification of two timestamps before and
after which data is filtered out. This allows a model where late data collected
after the intended analysis window can be included, and any late-arriving data
prior to the beginning of the analysis window can be removed as well. By using
windowing and adding element timestamps, we can do finer-grained analysis than
with the `user_score` pipeline. However, our batch processing is high-latency,
in that we don't get results from plays at the beginning of the batch's time
period until the batch is processed.

Optionally include the `--input` argument to specify a batch input file. To
indicate a time after which the data should be filtered out, include the
`--stop_min` arg. E.g., `--stop_min=2015-10-18-23-59` indicates that any data
timestamped after 23:59 PST on 2015-10-18 should not be included in the
analysis. To indicate a time before which data should be filtered out, include
the `--start_min` arg. If you're using the default input
"gs://dataflow-samples/game/gaming_data*.csv", then
`--start_min=2015-11-16-16-10 --stop_min=2015-11-17-16-10` are good values.

For a description of the usage and options, use -h or --help.

To specify a different runner:
  --runner YOUR_RUNNER

NOTE: When specifying a different runner, additional runner-specific options
      may have to be passed in as well

EXAMPLES
--------

# DirectRunner
python hourly_team_score.py     --project $PROJECT_ID     --dataset $BIGQUERY_DATASET

# DataflowRunner
python hourly_team_score.py     --project $PROJECT_ID     --region $REGION_ID     --dataset $BIGQUERY_DATASET     --runner DataflowRunner     --temp_location gs://$BUCKET/user_score/temp
"""
import argparse
import csv
import logging
import sys
import time
from datetime import datetime
import apache_beam as beam
from apache_beam.metrics.metric import Metrics
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

def str2timestamp(s, fmt='%Y-%m-%d-%H-%M'):
    if False:
        i = 10
        return i + 15
    'Converts a string into a unix timestamp.'
    dt = datetime.strptime(s, fmt)
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds()

def timestamp2str(t, fmt='%Y-%m-%d %H:%M:%S.000'):
    if False:
        return 10
    'Converts a unix timestamp into a formatted string.'
    return datetime.fromtimestamp(t).strftime(fmt)

class ParseGameEventFn(beam.DoFn):
    """Parses the raw game event info into a Python dictionary.

  Each event line has the following format:
    username,teamname,score,timestamp_in_ms,readable_time

  e.g.:
    user2_AsparagusPig,AsparagusPig,10,1445230923951,2015-11-02 09:09:28.224

  The human-readable time string is not used here.
  """

    def __init__(self):
        if False:
            return 10
        beam.DoFn.__init__(self)
        self.num_parse_errors = Metrics.counter(self.__class__, 'num_parse_errors')

    def process(self, elem):
        if False:
            while True:
                i = 10
        try:
            row = list(csv.reader([elem]))[0]
            yield {'user': row[0], 'team': row[1], 'score': int(row[2]), 'timestamp': int(row[3]) / 1000.0}
        except:
            self.num_parse_errors.inc()
            logging.error('Parse error on "%s"', elem)

class ExtractAndSumScore(beam.PTransform):
    """A transform to extract key/score information and sum the scores.
  The constructor argument `field` determines whether 'team' or 'user' info is
  extracted.
  """

    def __init__(self, field):
        if False:
            return 10
        beam.PTransform.__init__(self)
        self.field = field

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | beam.Map(lambda elem: (elem[self.field], elem['score'])) | beam.CombinePerKey(sum)

class TeamScoresDict(beam.DoFn):
    """Formats the data into a dictionary of BigQuery columns with their values

  Receives a (team, score) pair, extracts the window start timestamp, and
  formats everything together into a dictionary. The dictionary is in the format
  {'bigquery_column': value}
  """

    def process(self, team_score, window=beam.DoFn.WindowParam):
        if False:
            i = 10
            return i + 15
        (team, score) = team_score
        start = timestamp2str(int(window.start))
        yield {'team': team, 'total_score': score, 'window_start': start, 'processing_time': timestamp2str(int(time.time()))}

class WriteToBigQuery(beam.PTransform):
    """Generate, format, and write BigQuery table row information."""

    def __init__(self, table_name, dataset, schema, project):
        if False:
            for i in range(10):
                print('nop')
        "Initializes the transform.\n    Args:\n      table_name: Name of the BigQuery table to use.\n      dataset: Name of the dataset to use.\n      schema: Dictionary in the format {'column_name': 'bigquery_type'}\n      project: Name of the Cloud project containing BigQuery table.\n    "
        beam.PTransform.__init__(self)
        self.table_name = table_name
        self.dataset = dataset
        self.schema = schema
        self.project = project

    def get_schema(self):
        if False:
            while True:
                i = 10
        'Build the output table schema.'
        return ', '.join(('%s:%s' % (col, self.schema[col]) for col in self.schema))

    def expand(self, pcoll):
        if False:
            print('Hello World!')
        return pcoll | 'ConvertToRow' >> beam.Map(lambda elem: {col: elem[col] for col in self.schema}) | beam.io.WriteToBigQuery(self.table_name, self.dataset, self.project, self.get_schema())

class HourlyTeamScore(beam.PTransform):

    def __init__(self, start_min, stop_min, window_duration):
        if False:
            while True:
                i = 10
        beam.PTransform.__init__(self)
        self.start_timestamp = str2timestamp(start_min)
        self.stop_timestamp = str2timestamp(stop_min)
        self.window_duration_in_seconds = window_duration * 60

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | 'ParseGameEventFn' >> beam.ParDo(ParseGameEventFn()) | 'FilterStartTime' >> beam.Filter(lambda elem: elem['timestamp'] > self.start_timestamp) | 'FilterEndTime' >> beam.Filter(lambda elem: elem['timestamp'] < self.stop_timestamp) | 'AddEventTimestamps' >> beam.Map(lambda elem: beam.window.TimestampedValue(elem, elem['timestamp'])) | 'FixedWindowsTeam' >> beam.WindowInto(beam.window.FixedWindows(self.window_duration_in_seconds)) | 'ExtractAndSumScore' >> ExtractAndSumScore('team')

def run(argv=None, save_main_session=True):
    if False:
        i = 10
        return i + 15
    'Main entry point; defines and runs the hourly_team_score pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='gs://apache-beam-samples/game/gaming_data*.csv', help='Path to the data file(s) containing game data.')
    parser.add_argument('--dataset', type=str, required=True, help='BigQuery Dataset to write tables to. Must already exist.')
    parser.add_argument('--table_name', default='leader_board', help='The BigQuery table name. Should not already exist.')
    parser.add_argument('--window_duration', type=int, default=60, help='Numeric value of fixed window duration, in minutes')
    parser.add_argument('--start_min', type=str, default='1970-01-01-00-00', help="String representation of the first minute after which to generate results in the format: yyyy-MM-dd-HH-mm. Any input data timestamped prior to that minute won't be included in the sums.")
    parser.add_argument('--stop_min', type=str, default='2100-01-01-00-00', help="String representation of the first minute for which to generate results in the format: yyyy-MM-dd-HH-mm. Any input data timestamped after to that minute won't be included in the sums.")
    (args, pipeline_args) = parser.parse_known_args(argv)
    options = PipelineOptions(pipeline_args)
    if options.view_as(GoogleCloudOptions).project is None:
        parser.print_usage()
        print(sys.argv[0] + ': error: argument --project is required')
        sys.exit(1)
    options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=options) as p:
        p | 'ReadInputText' >> beam.io.ReadFromText(args.input) | 'HourlyTeamScore' >> HourlyTeamScore(args.start_min, args.stop_min, args.window_duration) | 'TeamScoresDict' >> beam.ParDo(TeamScoresDict()) | 'WriteTeamScoreSums' >> WriteToBigQuery(args.table_name, args.dataset, {'team': 'STRING', 'total_score': 'INTEGER', 'window_start': 'STRING'}, options.view_as(GoogleCloudOptions).project)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()