import argparse
import logging
import sys
from pyflink.table import TableEnvironment, EnvironmentSettings, TableDescriptor, Schema, DataTypes, FormatDescriptor
from pyflink.table.expressions import col, lit
from pyflink.table.udf import udf
words = ['flink', 'window', 'timer', 'event_time', 'processing_time', 'state', 'connector', 'pyflink', 'checkpoint', 'watermark', 'sideoutput', 'sql', 'datastream', 'broadcast', 'asyncio', 'catalog', 'batch', 'streaming']
max_word_id = len(words) - 1

def streaming_word_count(output_path):
    if False:
        i = 10
        return i + 15
    t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    t_env.create_temporary_table('source', TableDescriptor.for_connector('datagen').schema(Schema.new_builder().column('word_id', DataTypes.INT()).build()).option('fields.word_id.kind', 'random').option('fields.word_id.min', '0').option('fields.word_id.max', str(max_word_id)).option('rows-per-second', '5').build())
    tab = t_env.from_path('source')
    if output_path is not None:
        t_env.create_temporary_table('sink', TableDescriptor.for_connector('filesystem').schema(Schema.new_builder().column('word', DataTypes.STRING()).column('count', DataTypes.BIGINT()).build()).option('path', output_path).format(FormatDescriptor.for_format('canal-json').build()).build())
    else:
        print('Printing result to stdout. Use --output to specify output path.')
        t_env.create_temporary_table('sink', TableDescriptor.for_connector('print').schema(Schema.new_builder().column('word', DataTypes.STRING()).column('count', DataTypes.BIGINT()).build()).build())

    @udf(result_type='string')
    def id_to_word(word_id):
        if False:
            i = 10
            return i + 15
        return words[word_id]
    tab.select(id_to_word(col('word_id'))).alias('word').group_by(col('word')).select(col('word'), lit(1).count).execute_insert('sink').wait()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', dest='output', required=False, help='Output file to write results to.')
    argv = sys.argv[1:]
    (known_args, _) = parser.parse_known_args(argv)
    streaming_word_count(known_args.output)