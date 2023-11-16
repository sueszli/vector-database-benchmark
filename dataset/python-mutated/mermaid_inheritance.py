"""
    mermaid_inheritance
    ~~~~~~~~~~~~~~~~~~~

    Modified by the CoSApp team from sphinx.ext.inheritance_diagram
    https://gitlab.com/cosapp/cosapp

    Defines a docutils directive for inserting inheritance diagrams.
    Provide the directive with one or more classes or modules (separated
    by whitespace).  For modules, all of the classes in that module will
    be used.
    Example::
       Given the following classes:
       class A: pass
       class B(A): pass
       class C(A): pass
       class D(B, C): pass
       class E(B): pass
       .. inheritance-diagram: D E
       Produces a graph like the following:
                   A
                  / \\
                 B   C
                / \\ /
               E   D
    The graph is inserted as a PNG+image map into HTML and a PDF in
    LaTeX.
    :copyright: Copyright 2007-2020 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from typing import Any, Dict, Iterable, List, cast
import sphinx
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
from mermaid import render_mm_html, render_mm_latex, render_mm_texinfo
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.inheritance_diagram import InheritanceDiagram, InheritanceException, InheritanceGraph, figure_wrapper, get_graph_hash, inheritance_diagram, skip
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator

class MermaidGraph(InheritanceGraph):
    """
    Given a list of classes, determines the set of classes that they inherit
    from all the way to the root "object", and then is able to generate a
    mermaid graph from them.
    """
    default_graph_attrs = {}
    default_node_attrs = {}
    default_edge_attrs = {}

    def _format_node_attrs(self, attrs: Dict) -> str:
        if False:
            print('Hello World!')
        return ''

    def _format_graph_attrs(self, attrs: Dict) -> str:
        if False:
            return 10
        return ''

    def generate_dot(self, name: str, urls: Dict={}, env: BuildEnvironment=None, graph_attrs: Dict={}, node_attrs: Dict={}, edge_attrs: Dict={}) -> str:
        if False:
            while True:
                i = 10
        'Generate a mermaid graph from the classes that were passed in\n        to __init__.\n        *name* is the name of the graph.\n        *urls* is a dictionary mapping class names to HTTP URLs.\n        *graph_attrs*, *node_attrs*, *edge_attrs* are dictionaries containing\n        key/value pairs to pass on as graphviz properties.\n        '
        res = []
        res.append('classDiagram\n')
        for (name, fullname, bases, tooltip) in sorted(self.class_info):
            res.append('  class {!s}\n'.format(name))
            if fullname in urls:
                res.append('  link {!s} "./{!s}" {!s}\n'.format(name, urls[fullname], tooltip or '"{}"'.format(name)))
            for base_name in bases:
                res.append('  {!s} <|-- {!s}\n'.format(base_name, name))
        return ''.join(res)

class mermaid_inheritance(inheritance_diagram):
    """
    A docutils node to use as a placeholder for the inheritance diagram.
    """
    pass

class MermaidDiagram(InheritanceDiagram):
    """
    Run when the mermaid_inheritance directive is first encountered.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'parts': int, 'private-bases': directives.flag, 'caption': directives.unchanged, 'top-classes': directives.unchanged_required}

    def run(self) -> List[Node]:
        if False:
            i = 10
            return i + 15
        node = mermaid_inheritance()
        node.document = self.state.document
        class_names = self.arguments[0].split()
        class_role = self.env.get_domain('py').role('class')
        node['parts'] = self.options.get('parts', 0)
        node['content'] = ', '.join(class_names)
        node['top-classes'] = []
        for cls in self.options.get('top-classes', '').split(','):
            cls = cls.strip()
            if cls:
                node['top-classes'].append(cls)
        try:
            graph = MermaidGraph(class_names, self.env.ref_context.get('py:module'), parts=node['parts'], private_bases='private-bases' in self.options, aliases=self.config.inheritance_alias, top_classes=node['top-classes'])
        except InheritanceException as err:
            return [node.document.reporter.warning(err, line=self.lineno)]
        for name in graph.get_all_class_names():
            (refnodes, x) = class_role('class', ':class:`%s`' % name, name, 0, self.state)
            node.extend(refnodes)
        node['graph'] = graph
        if 'caption' not in self.options:
            self.add_name(node)
            return [node]
        else:
            figure = figure_wrapper(self, node, self.options['caption'])
            self.add_name(figure)
            return [figure]

def html_visit_mermaid_inheritance(self: HTMLTranslator, node: inheritance_diagram) -> None:
    if False:
        print('Hello World!')
    '\n    Output the graph for HTML.  This will insert a PNG with clickable\n    image map.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    mermaid_output_format = self.builder.env.config.mermaid_output_format.upper()
    current_filename = self.builder.current_docname + self.builder.out_suffix
    urls = {}
    pending_xrefs = cast(Iterable[addnodes.pending_xref], node)
    for child in pending_xrefs:
        if child.get('refuri') is not None:
            if mermaid_output_format == 'SVG':
                urls[child['reftitle']] = '../' + child.get('refuri')
            else:
                urls[child['reftitle']] = child.get('refuri')
        elif child.get('refid') is not None:
            if mermaid_output_format == 'SVG':
                urls[child['reftitle']] = '../' + current_filename + '#' + child.get('refid')
            else:
                urls[child['reftitle']] = '#' + child.get('refid')
    dotcode = graph.generate_dot(name, urls, env=self.builder.env)
    render_mm_html(self, node, dotcode, {}, 'inheritance', 'inheritance', alt='Inheritance diagram of ' + node['content'])
    raise nodes.SkipNode

def latex_visit_mermaid_inheritance(self: LaTeXTranslator, node: inheritance_diagram) -> None:
    if False:
        while True:
            i = 10
    '\n    Output the graph for LaTeX.  This will insert a PDF.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    dotcode = graph.generate_dot(name, env=self.builder.env)
    render_mm_latex(self, node, dotcode, {}, 'inheritance')
    raise nodes.SkipNode

def texinfo_visit_mermaid_inheritance(self: TexinfoTranslator, node: inheritance_diagram) -> None:
    if False:
        while True:
            i = 10
    '\n    Output the graph for Texinfo.  This will insert a PNG.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    dotcode = graph.generate_dot(name, env=self.builder.env)
    render_mm_texinfo(self, node, dotcode, {}, 'inheritance')
    raise nodes.SkipNode

def setup(app: Sphinx) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    app.setup_extension('mermaid')
    app.add_node(mermaid_inheritance, latex=(latex_visit_mermaid_inheritance, None), html=(html_visit_mermaid_inheritance, None), text=(skip, None), man=(skip, None), texinfo=(texinfo_visit_mermaid_inheritance, None))
    app.add_directive('mermaid-inheritance', MermaidDiagram)
    app.add_config_value('inheritance_alias', {}, False)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}