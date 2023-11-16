"""An example of using custom classes and coder for grouping operations.

This workflow demonstrates registration and usage of a custom coder for a user-
defined class. A deterministic custom coder is needed to use a class as a key in
a combine or group operation.

This example assumes an input file with, on each line, a comma-separated name
and score.
"""
import argparse
import logging
import sys
import typing
import apache_beam as beam
from apache_beam import coders
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.typehints.decorators import with_output_types

class Player(object):
    """A custom class used as a key in combine/group transforms."""

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

class PlayerCoder(coders.Coder):
    """A custom coder for the Player class."""

    def encode(self, o):
        if False:
            while True:
                i = 10
        'Encode to bytes with a trace that coder was used.'
        return b'x:%s' % str(o.name).encode('utf-8')

    def decode(self, s):
        if False:
            while True:
                i = 10
        s = s.decode('utf-8')
        assert s[0:2] == 'x:'
        return Player(s[2:])

    def is_deterministic(self):
        if False:
            while True:
                i = 10
        return True

@with_output_types(typing.Tuple[Player, int])
def get_players(descriptor):
    if False:
        print('Hello World!')
    (name, points) = descriptor.split(',')
    return (Player(name), int(points))

def run(args=None, save_main_session=True):
    if False:
        for i in range(10):
            print('nop')
    'Runs the workflow computing total points from a collection of matches.'
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input file to process.')
    parser.add_argument('--output', required=True, help='Output file to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args(args)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        coders.registry.register_coder(Player, PlayerCoder)
        p | ReadFromText(known_args.input) | beam.Map(get_players) | beam.CombinePerKey(sum) | beam.Map(lambda k_v: '%s,%d' % (k_v[0].name, k_v[1])) | WriteToText(known_args.output)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()