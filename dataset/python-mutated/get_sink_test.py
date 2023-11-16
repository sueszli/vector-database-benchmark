import get_sink
import os
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_get_sink():
    if False:
        i = 10
        return i + 15
    sink_name = '_Default'
    sink = get_sink.get_sink(PROJECT, sink_name)
    assert sink_name in sink.destination