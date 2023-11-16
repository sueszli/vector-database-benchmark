from __future__ import print_function
from __future__ import unicode_literals
from builtins import str, bytes, dict, int
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pattern.graph import Graph, CSS, CANVAS
template = '\n<!doctype html> \n<html>\n<head>\n\t<meta charset="utf-8">\n\t<script type="text/javascript" src="canvas.js"></script>\n\t<script type="text/javascript" src="graph.js"></script>\n\t<style type="text/css">\n\t\t%s\n\t</style>\n</head>\n<body> \n\t%s\n</body>\n</html>\n'.strip()

def webpage(graph, **kwargs):
    if False:
        while True:
            i = 10
    s1 = graph.serialize(CSS, **kwargs)
    s2 = graph.serialize(CANVAS, **kwargs)
    return template % (s1.replace('\n', '\n\t\t'), s2.replace('\n', '\n\t'))
g = Graph()
g.add_node('cat')
g.add_node('dog')
g.add_edge('cat', 'dog')
print(webpage(g, width=500, height=500))