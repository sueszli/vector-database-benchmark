from os import path
import sys
from grc.converter import flow_graph

def test_flow_graph_converter():
    if False:
        while True:
            i = 10
    filename = path.join(path.dirname(__file__), 'resources', 'test_compiler.grc')
    data = flow_graph.from_xml(filename)
    flow_graph.dump(data, sys.stdout)

def test_flow_graph_converter_with_fp():
    if False:
        print('Hello World!')
    filename = path.join(path.dirname(__file__), 'resources', 'test_compiler.grc')
    with open(filename, 'rb') as fp:
        data = flow_graph.from_xml(fp)
    flow_graph.dump(data, sys.stdout)