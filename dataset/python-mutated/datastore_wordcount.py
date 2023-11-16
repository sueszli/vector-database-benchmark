"""A word-counting workflow that uses Google Cloud Datastore.

This example shows how to use ``datastoreio`` to read from and write to
Google Cloud Datastore. Note that running this example may incur charge for
Cloud Datastore operations.

See https://developers.google.com/datastore/ for more details on Google Cloud
Datastore.
See https://beam.apache.org/get-started/quickstart on
how to run a Beam pipeline.

Read-only Mode: In this mode, this example reads Cloud Datastore entities using
the ``datastoreio.ReadFromDatastore`` transform, extracts the words,
counts them and write the output to a set of files.

The following options must be provided to run this pipeline in read-only mode:
``
--project GCP_PROJECT
--kind YOUR_DATASTORE_KIND
--output [YOUR_LOCAL_FILE *or* gs://YOUR_OUTPUT_PATH]
--read_only
``

Read-write Mode: In this mode, this example reads words from an input file,
converts them to Beam ``Entity`` objects and writes them to Cloud Datastore
using the ``datastoreio.WriteToDatastore`` transform. The second pipeline
will then read these Cloud Datastore entities using the
``datastoreio.ReadFromDatastore`` transform, extract the words, count them and
write the output to a set of files.

The following options must be provided to run this pipeline in read-write mode:
``
--project GCP_PROJECT
--kind YOUR_DATASTORE_KIND
--output [YOUR_LOCAL_FILE *or* gs://YOUR_OUTPUT_PATH]
``
"""
import argparse
import logging
import re
import sys
from typing import Iterable
from typing import Optional
from typing import Text
import uuid
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io.gcp.datastore.v1new.datastoreio import ReadFromDatastore
from apache_beam.io.gcp.datastore.v1new.datastoreio import WriteToDatastore
from apache_beam.io.gcp.datastore.v1new.types import Entity
from apache_beam.io.gcp.datastore.v1new.types import Key
from apache_beam.io.gcp.datastore.v1new.types import Query
from apache_beam.metrics import Metrics
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions

@beam.typehints.with_input_types(Entity)
@beam.typehints.with_output_types(Text)
class WordExtractingDoFn(beam.DoFn):
    """Parse each line of input text into words."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.empty_line_counter = Metrics.counter('main', 'empty_lines')
        self.word_length_counter = Metrics.counter('main', 'word_lengths')
        self.word_counter = Metrics.counter('main', 'total_words')
        self.word_lengths_dist = Metrics.distribution('main', 'word_len_dist')

    def process(self, element):
        if False:
            for i in range(10):
                print('nop')
        "Extract words from the 'content' property of Cloud Datastore entities.\n\n    The element is a line of text.  If the line is blank, note that, too.\n    Args:\n      element: the input entity to be processed\n    Returns:\n      A list of words found.\n    "
        text_line = element.properties.get('content', '')
        if not text_line:
            self.empty_line_counter.inc()
            return None
        words = re.findall("[A-Za-z\\']+", text_line)
        for w in words:
            self.word_length_counter.inc(len(w))
            self.word_lengths_dist.update(len(w))
            self.word_counter.inc()
        return words

class EntityWrapper(object):
    """Create a Cloud Datastore entity from the given string."""

    def __init__(self, project, namespace, kind, ancestor):
        if False:
            return 10
        self._project = project
        self._namespace = namespace
        self._kind = kind
        self._ancestor = ancestor

    def make_entity(self, content):
        if False:
            while True:
                i = 10
        ancestor_key = Key([self._kind, self._ancestor], namespace=self._namespace, project=self._project)
        key = Key([self._kind, str(uuid.uuid4())], parent=ancestor_key)
        entity = Entity(key)
        entity.set_properties({'content': content})
        return entity

def write_to_datastore(project, user_options, pipeline_options):
    if False:
        for i in range(10):
            print('nop')
    'Creates a pipeline that writes entities to Cloud Datastore.'
    with beam.Pipeline(options=pipeline_options) as p:
        _ = p | 'read' >> ReadFromText(user_options.input) | 'create entity' >> beam.Map(EntityWrapper(project, user_options.namespace, user_options.kind, user_options.ancestor).make_entity) | 'write to datastore' >> WriteToDatastore(project)

def make_ancestor_query(project, kind, namespace, ancestor):
    if False:
        for i in range(10):
            print('nop')
    'Creates a Cloud Datastore ancestor query.\n\n  The returned query will fetch all the entities that have the parent key name\n  set to the given `ancestor`.\n  '
    ancestor_key = Key([kind, ancestor], project=project, namespace=namespace)
    return Query(kind, project, namespace, ancestor_key)

def read_from_datastore(project, user_options, pipeline_options):
    if False:
        print('Hello World!')
    'Creates a pipeline that reads entities from Cloud Datastore.'
    p = beam.Pipeline(options=pipeline_options)
    query = make_ancestor_query(project, user_options.kind, user_options.namespace, user_options.ancestor)
    lines = p | 'read from datastore' >> ReadFromDatastore(query)

    def count_ones(word_ones):
        if False:
            for i in range(10):
                print('nop')
        (word, ones) = word_ones
        return (word, sum(ones))
    counts = lines | 'split' >> beam.ParDo(WordExtractingDoFn()) | 'pair_with_one' >> beam.Map(lambda x: (x, 1)) | 'group' >> beam.GroupByKey() | 'count' >> beam.Map(count_ones)

    def format_result(word_count):
        if False:
            return 10
        (word, count) = word_count
        return '%s: %s' % (word, count)
    output = counts | 'format' >> beam.Map(format_result)
    output | 'write' >> beam.io.WriteToText(file_path_prefix=user_options.output, num_shards=user_options.num_shards)
    result = p.run()
    result.wait_until_finish()
    return result

def run(argv=None):
    if False:
        for i in range(10):
            print('nop')
    'Main entry point; defines and runs the wordcount pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--kind', dest='kind', required=True, help='Datastore Kind')
    parser.add_argument('--namespace', dest='namespace', help='Datastore Namespace')
    parser.add_argument('--ancestor', dest='ancestor', default='root', help='The ancestor key name for all entities.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    parser.add_argument('--read_only', action='store_true', help='Read an existing dataset, do not write first')
    parser.add_argument('--num_shards', dest='num_shards', type=int, default=0, help='Number of output shards')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    project = pipeline_options.view_as(GoogleCloudOptions).project
    if project is None:
        parser.print_usage()
        print(sys.argv[0] + ': error: argument --project is required')
        sys.exit(1)
    if not known_args.read_only:
        write_to_datastore(project, known_args, pipeline_options)
    result = read_from_datastore(project, known_args, pipeline_options)
    empty_lines_filter = MetricsFilter().with_name('empty_lines')
    query_result = result.metrics().query(empty_lines_filter)
    if query_result['counters']:
        empty_lines_counter = query_result['counters'][0]
        logging.info('number of empty lines: %d', empty_lines_counter.committed)
    else:
        logging.warning('unable to retrieve counter metrics from runner')
    word_lengths_filter = MetricsFilter().with_name('word_len_dist')
    query_result = result.metrics().query(word_lengths_filter)
    if query_result['distributions']:
        word_lengths_dist = query_result['distributions'][0]
        logging.info('average word length: %d', word_lengths_dist.committed.mean)
    else:
        logging.warning('unable to retrieve distribution metrics from runner')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()