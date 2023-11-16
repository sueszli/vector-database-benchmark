"""Renders parse trees with Graphviz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import warnings
import pygraphviz

def parse_tree_graph(sentence):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a parse tree graph.\n\n  Args:\n    sentence: syntaxnet.Sentence instance.\n\n  Returns:\n    HTML graph contents, as a string.\n  '
    graph = pygraphviz.AGraph(directed=True, strict=False, rankdir='TB')
    for (i, token) in enumerate(sentence.token):
        node_id = 'tok_{}'.format(i)
        graph.add_node(node_id, label=token.word)
        if token.head >= 0:
            src_id = 'tok_{}'.format(token.head)
            graph.add_edge(src_id, node_id, label=token.label, key='parse_{}_{}'.format(node_id, src_id))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        svg = graph.draw(format='svg', prog='dot')
    svg = unicode(svg, 'utf-8')
    image_and_text = u'<p><em>Text:</em> {}</p>{}'.format(' '.join((token.word for token in sentence.token)), svg)
    new_window_html = (u"<style type='text/css'>svg { max-width: 100%; }</style>" + image_and_text).encode('utf-8')
    as_uri = 'data:text/html;charset=utf-8;base64,{}'.format(base64.b64encode(new_window_html))
    return u"{}<p><a target='_blank' href='{}'>Open in new window</a></p>".format(image_and_text, as_uri)