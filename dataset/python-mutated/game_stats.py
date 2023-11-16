"""Fourth in a series of four pipelines that tell a story in a 'gaming' domain.

New concepts: session windows and finding session duration; use of both
singleton and non-singleton side inputs.

This pipeline builds on the {@link LeaderBoard} functionality, and adds some
"business intelligence" analysis: abuse detection and usage patterns. The
pipeline derives the Mean user score sum for a window, and uses that information
to identify likely spammers/robots. (The robots have a higher click rate than
the human users). The 'robot' users are then filtered out when calculating the
team scores.

Additionally, user sessions are tracked: that is, we find bursts of user
activity using session windows. Then, the mean session duration information is
recorded in the context of subsequent fixed windowing. (This could be used to
tell us what games are giving us greater user retention).

Run injector.Injector to generate pubsub data for this pipeline. The Injector
documentation provides more detail on how to do this. The injector is currently
implemented in Java only, it can be used from the Java SDK.

The PubSub topic you specify should be the same topic to which the Injector is
publishing.

To run the Java injector:
<beam_root>/examples/java$ mvn compile exec:java     -Dexec.mainClass=org.apache.beam.examples.complete.game.injector.Injector     -Dexec.args="$PROJECT_ID $PUBSUB_TOPIC none"

For a description of the usage and options, use -h or --help.

To specify a different runner:
  --runner YOUR_RUNNER

NOTE: When specifying a different runner, additional runner-specific options
      may have to be passed in as well

EXAMPLES
--------

# DirectRunner
python game_stats.py     --project $PROJECT_ID     --topic projects/$PROJECT_ID/topics/$PUBSUB_TOPIC     --dataset $BIGQUERY_DATASET

# DataflowRunner
python game_stats.py     --project $PROJECT_ID     --region $REGION_ID     --topic projects/$PROJECT_ID/topics/$PUBSUB_TOPIC     --dataset $BIGQUERY_DATASET     --runner DataflowRunner     --temp_location gs://$BUCKET/user_score/temp
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
from apache_beam.options.pipeline_options import StandardOptions

def timestamp2str(t, fmt='%Y-%m-%d %H:%M:%S.000'):
    if False:
        print('Hello World!')
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
            print('Hello World!')
        beam.DoFn.__init__(self)
        self.num_parse_errors = Metrics.counter(self.__class__, 'num_parse_errors')

    def process(self, elem):
        if False:
            return 10
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
            while True:
                i = 10
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
            return 10
        (team, score) = team_score
        start = timestamp2str(int(window.start))
        yield {'team': team, 'total_score': score, 'window_start': start, 'processing_time': timestamp2str(int(time.time()))}

class WriteToBigQuery(beam.PTransform):
    """Generate, format, and write BigQuery table row information."""

    def __init__(self, table_name, dataset, schema, project):
        if False:
            return 10
        "Initializes the transform.\n    Args:\n      table_name: Name of the BigQuery table to use.\n      dataset: Name of the dataset to use.\n      schema: Dictionary in the format {'column_name': 'bigquery_type'}\n      project: Name of the Cloud project containing BigQuery table.\n    "
        beam.PTransform.__init__(self)
        self.table_name = table_name
        self.dataset = dataset
        self.schema = schema
        self.project = project

    def get_schema(self):
        if False:
            for i in range(10):
                print('nop')
        'Build the output table schema.'
        return ', '.join(('%s:%s' % (col, self.schema[col]) for col in self.schema))

    def expand(self, pcoll):
        if False:
            return 10
        return pcoll | 'ConvertToRow' >> beam.Map(lambda elem: {col: elem[col] for col in self.schema}) | beam.io.WriteToBigQuery(self.table_name, self.dataset, self.project, self.get_schema())

class CalculateSpammyUsers(beam.PTransform):
    """Filter out all but those users with a high clickrate, which we will
  consider as 'spammy' uesrs.

  We do this by finding the mean total score per user, then using that
  information as a side input to filter out all but those user scores that are
  larger than (mean * SCORE_WEIGHT).
  """
    SCORE_WEIGHT = 2.5

    def expand(self, user_scores):
        if False:
            print('Hello World!')
        sum_scores = user_scores | 'SumUsersScores' >> beam.CombinePerKey(sum)
        global_mean_score = sum_scores | beam.Values() | beam.CombineGlobally(beam.combiners.MeanCombineFn()).as_singleton_view()
        filtered = sum_scores | 'ProcessAndFilter' >> beam.Filter(lambda key_score, global_mean: key_score[1] > global_mean * self.SCORE_WEIGHT, global_mean_score)
        return filtered

class UserSessionActivity(beam.DoFn):
    """Calculate and output an element's session duration, in seconds."""

    def process(self, elem, window=beam.DoFn.WindowParam):
        if False:
            return 10
        yield ((window.end.micros - window.start.micros) // 1000000)

def run(argv=None, save_main_session=True):
    if False:
        for i in range(10):
            print('nop')
    'Main entry point; defines and runs the hourly_team_score pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, help='Pub/Sub topic to read from')
    parser.add_argument('--subscription', type=str, help='Pub/Sub subscription to read from')
    parser.add_argument('--dataset', type=str, required=True, help='BigQuery Dataset to write tables to. Must already exist.')
    parser.add_argument('--table_name', type=str, default='game_stats', help='The BigQuery table name. Should not already exist.')
    parser.add_argument('--fixed_window_duration', type=int, default=60, help='Numeric value of fixed window duration for user analysis, in minutes')
    parser.add_argument('--session_gap', type=int, default=5, help='Numeric value of gap between user sessions, in minutes')
    parser.add_argument('--user_activity_window_duration', type=int, default=30, help='Numeric value of fixed window for finding mean of user session duration, in minutes')
    (args, pipeline_args) = parser.parse_known_args(argv)
    if args.topic is None and args.subscription is None:
        parser.print_usage()
        print(sys.argv[0] + ': error: one of --topic or --subscription is required')
        sys.exit(1)
    options = PipelineOptions(pipeline_args)
    if options.view_as(GoogleCloudOptions).project is None:
        parser.print_usage()
        print(sys.argv[0] + ': error: argument --project is required')
        sys.exit(1)
    fixed_window_duration = args.fixed_window_duration * 60
    session_gap = args.session_gap * 60
    user_activity_window_duration = args.user_activity_window_duration * 60
    options.view_as(SetupOptions).save_main_session = save_main_session
    options.view_as(StandardOptions).streaming = True
    with beam.Pipeline(options=options) as p:
        if args.subscription:
            scores = p | 'ReadPubSub' >> beam.io.ReadFromPubSub(subscription=args.subscription)
        else:
            scores = p | 'ReadPubSub' >> beam.io.ReadFromPubSub(topic=args.topic)
        raw_events = scores | 'DecodeString' >> beam.Map(lambda b: b.decode('utf-8')) | 'ParseGameEventFn' >> beam.ParDo(ParseGameEventFn()) | 'AddEventTimestamps' >> beam.Map(lambda elem: beam.window.TimestampedValue(elem, elem['timestamp']))
        user_events = raw_events | 'ExtractUserScores' >> beam.Map(lambda elem: (elem['user'], elem['score']))
        spammers_view = user_events | 'UserFixedWindows' >> beam.WindowInto(beam.window.FixedWindows(fixed_window_duration)) | 'CalculateSpammyUsers' >> CalculateSpammyUsers() | 'CreateSpammersView' >> beam.CombineGlobally(beam.combiners.ToDictCombineFn()).as_singleton_view()
        raw_events | 'WindowIntoFixedWindows' >> beam.WindowInto(beam.window.FixedWindows(fixed_window_duration)) | 'FilterOutSpammers' >> beam.Filter(lambda elem, spammers: elem['user'] not in spammers, spammers_view) | 'ExtractAndSumScore' >> ExtractAndSumScore('team') | 'TeamScoresDict' >> beam.ParDo(TeamScoresDict()) | 'WriteTeamScoreSums' >> WriteToBigQuery(args.table_name + '_teams', args.dataset, {'team': 'STRING', 'total_score': 'INTEGER', 'window_start': 'STRING', 'processing_time': 'STRING'}, options.view_as(GoogleCloudOptions).project)
        user_events | 'WindowIntoSessions' >> beam.WindowInto(beam.window.Sessions(session_gap), timestamp_combiner=beam.window.TimestampCombiner.OUTPUT_AT_EOW) | beam.CombinePerKey(lambda _: None) | 'UserSessionActivity' >> beam.ParDo(UserSessionActivity()) | 'WindowToExtractSessionMean' >> beam.WindowInto(beam.window.FixedWindows(user_activity_window_duration)) | beam.CombineGlobally(beam.combiners.MeanCombineFn()).without_defaults() | 'FormatAvgSessionLength' >> beam.Map(lambda elem: {'mean_duration': float(elem)}) | 'WriteAvgSessionLength' >> WriteToBigQuery(args.table_name + '_sessions', args.dataset, {'mean_duration': 'FLOAT'}, options.view_as(GoogleCloudOptions).project)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()