""" Provide a base class and useful functions for Bokeh Sphinx directives.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import re
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
py_sig_re = re.compile('^ ([\\w.]*\\.)?            # class name(s)\n          (\\w+)  \\s*             # thing name\n          (?: \\((.*)\\)           # optional: arguments\n           (?:\\s* -> \\s* (.*))?  # return annotation\n          )? $                   # and nothing more\n          ', re.VERBOSE)
__all__ = ('BokehDirective', 'py_sig_re')

class BokehDirective(SphinxDirective):

    def parse(self, rst_text, annotation):
        if False:
            i = 10
            return i + 15
        result = ViewList()
        for line in rst_text.split('\n'):
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children