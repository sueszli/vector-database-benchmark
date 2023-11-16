"""Defines a docutils directive for inserting inheritance diagrams.

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
"""
from __future__ import annotations
import builtins
import hashlib
import inspect
import re
from collections.abc import Iterable, Sequence
from importlib import import_module
from os import path
from typing import TYPE_CHECKING, Any, cast
from docutils import nodes
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.ext.graphviz import figure_wrapper, graphviz, render_dot_html, render_dot_latex, render_dot_texinfo
from sphinx.util.docutils import SphinxDirective
if TYPE_CHECKING:
    from docutils.nodes import Node
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment
    from sphinx.util.typing import OptionSpec
    from sphinx.writers.html import HTML5Translator
    from sphinx.writers.latex import LaTeXTranslator
    from sphinx.writers.texinfo import TexinfoTranslator
module_sig_re = re.compile('^(?:([\\w.]*)\\.)?  # module names\n                           (\\w+)  \\s* $          # class/final module name\n                           ', re.VERBOSE)
py_builtins = [obj for obj in vars(builtins).values() if inspect.isclass(obj)]

def try_import(objname: str) -> Any:
    if False:
        i = 10
        return i + 15
    'Import a object or module using *name* and *currentmodule*.\n    *name* should be a relative name from *currentmodule* or\n    a fully-qualified name.\n\n    Returns imported object or module.  If failed, returns None value.\n    '
    try:
        return import_module(objname)
    except TypeError:
        return None
    except ImportError:
        matched = module_sig_re.match(objname)
        if not matched:
            return None
        (modname, attrname) = matched.groups()
        if modname is None:
            return None
        try:
            module = import_module(modname)
            return getattr(module, attrname, None)
        except ImportError:
            return None

def import_classes(name: str, currmodule: str) -> Any:
    if False:
        print('Hello World!')
    'Import a class using its fully-qualified *name*.'
    target = None
    if currmodule:
        target = try_import(currmodule + '.' + name)
    if target is None:
        target = try_import(name)
    if target is None:
        raise InheritanceException('Could not import class or module %r specified for inheritance diagram' % name)
    if inspect.isclass(target):
        return [target]
    elif inspect.ismodule(target):
        classes = []
        for cls in target.__dict__.values():
            if inspect.isclass(cls) and cls.__module__ == target.__name__:
                classes.append(cls)
        return classes
    raise InheritanceException('%r specified for inheritance diagram is not a class or module' % name)

class InheritanceException(Exception):
    pass

class InheritanceGraph:
    """
    Given a list of classes, determines the set of classes that they inherit
    from all the way to the root "object", and then is able to generate a
    graphviz dot graph from them.
    """

    def __init__(self, class_names: list[str], currmodule: str, show_builtins: bool=False, private_bases: bool=False, parts: int=0, aliases: dict[str, str] | None=None, top_classes: Sequence[Any]=()) -> None:
        if False:
            return 10
        '*class_names* is a list of child classes to show bases from.\n\n        If *show_builtins* is True, then Python builtins will be shown\n        in the graph.\n        '
        self.class_names = class_names
        classes = self._import_classes(class_names, currmodule)
        self.class_info = self._class_info(classes, show_builtins, private_bases, parts, aliases, top_classes)
        if not self.class_info:
            msg = 'No classes found for inheritance diagram'
            raise InheritanceException(msg)

    def _import_classes(self, class_names: list[str], currmodule: str) -> list[Any]:
        if False:
            print('Hello World!')
        'Import a list of classes.'
        classes: list[Any] = []
        for name in class_names:
            classes.extend(import_classes(name, currmodule))
        return classes

    def _class_info(self, classes: list[Any], show_builtins: bool, private_bases: bool, parts: int, aliases: dict[str, str] | None, top_classes: Sequence[Any]) -> list[tuple[str, str, list[str], str]]:
        if False:
            while True:
                i = 10
        'Return name and bases for all classes that are ancestors of\n        *classes*.\n\n        *parts* gives the number of dotted name parts to include in the\n        displayed node names, from right to left. If given as a negative, the\n        number of parts to drop from the left. A value of 0 displays the full\n        dotted name. E.g. ``sphinx.ext.inheritance_diagram.InheritanceGraph``\n        with ``parts=2`` or ``parts=-2`` gets displayed as\n        ``inheritance_diagram.InheritanceGraph``, and as\n        ``ext.inheritance_diagram.InheritanceGraph`` with ``parts=3`` or\n        ``parts=-1``.\n\n        *top_classes* gives the name(s) of the top most ancestor class to\n        traverse to. Multiple names can be specified separated by comma.\n        '
        all_classes = {}

        def recurse(cls: Any) -> None:
            if False:
                i = 10
                return i + 15
            if not show_builtins and cls in py_builtins:
                return
            if not private_bases and cls.__name__.startswith('_'):
                return
            nodename = self.class_name(cls, parts, aliases)
            fullname = self.class_name(cls, 0, aliases)
            tooltip = None
            try:
                if cls.__doc__:
                    doc = cls.__doc__.strip().split('\n')[0]
                    if doc:
                        tooltip = '"%s"' % doc.replace('"', '\\"')
            except Exception:
                pass
            baselist: list[str] = []
            all_classes[cls] = (nodename, fullname, baselist, tooltip)
            if fullname in top_classes:
                return
            for base in cls.__bases__:
                if not show_builtins and base in py_builtins:
                    continue
                if not private_bases and base.__name__.startswith('_'):
                    continue
                baselist.append(self.class_name(base, parts, aliases))
                if base not in all_classes:
                    recurse(base)
        for cls in classes:
            recurse(cls)
        return list(all_classes.values())

    def class_name(self, cls: Any, parts: int=0, aliases: dict[str, str] | None=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Given a class object, return a fully-qualified name.\n\n        This works for things I've tested in matplotlib so far, but may not be\n        completely general.\n        "
        module = cls.__module__
        if module in ('__builtin__', 'builtins'):
            fullname = cls.__name__
        else:
            fullname = f'{module}.{cls.__qualname__}'
        if parts == 0:
            result = fullname
        else:
            name_parts = fullname.split('.')
            result = '.'.join(name_parts[-parts:])
        if aliases is not None and result in aliases:
            return aliases[result]
        return result

    def get_all_class_names(self) -> list[str]:
        if False:
            print('Hello World!')
        'Get all of the class names involved in the graph.'
        return [fullname for (_, fullname, _, _) in self.class_info]
    default_graph_attrs = {'rankdir': 'LR', 'size': '"8.0, 12.0"', 'bgcolor': 'transparent'}
    default_node_attrs = {'shape': 'box', 'fontsize': 10, 'height': 0.25, 'fontname': '"Vera Sans, DejaVu Sans, Liberation Sans, Arial, Helvetica, sans"', 'style': '"setlinewidth(0.5),filled"', 'fillcolor': 'white'}
    default_edge_attrs = {'arrowsize': 0.5, 'style': '"setlinewidth(0.5)"'}

    def _format_node_attrs(self, attrs: dict[str, Any]) -> str:
        if False:
            while True:
                i = 10
        return ','.join(['%s=%s' % x for x in sorted(attrs.items())])

    def _format_graph_attrs(self, attrs: dict[str, Any]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ''.join(['%s=%s;\n' % x for x in sorted(attrs.items())])

    def generate_dot(self, name: str, urls: dict[str, str] | None=None, env: BuildEnvironment | None=None, graph_attrs: dict | None=None, node_attrs: dict | None=None, edge_attrs: dict | None=None) -> str:
        if False:
            i = 10
            return i + 15
        'Generate a graphviz dot graph from the classes that were passed in\n        to __init__.\n\n        *name* is the name of the graph.\n\n        *urls* is a dictionary mapping class names to HTTP URLs.\n\n        *graph_attrs*, *node_attrs*, *edge_attrs* are dictionaries containing\n        key/value pairs to pass on as graphviz properties.\n        '
        if urls is None:
            urls = {}
        g_attrs = self.default_graph_attrs.copy()
        n_attrs = self.default_node_attrs.copy()
        e_attrs = self.default_edge_attrs.copy()
        if graph_attrs is not None:
            g_attrs.update(graph_attrs)
        if node_attrs is not None:
            n_attrs.update(node_attrs)
        if edge_attrs is not None:
            e_attrs.update(edge_attrs)
        if env:
            g_attrs.update(env.config.inheritance_graph_attrs)
            n_attrs.update(env.config.inheritance_node_attrs)
            e_attrs.update(env.config.inheritance_edge_attrs)
        res: list[str] = []
        res.append('digraph %s {\n' % name)
        res.append(self._format_graph_attrs(g_attrs))
        for (name, fullname, bases, tooltip) in sorted(self.class_info):
            this_node_attrs = n_attrs.copy()
            if fullname in urls:
                this_node_attrs['URL'] = '"%s"' % urls[fullname]
                this_node_attrs['target'] = '"_top"'
            if tooltip:
                this_node_attrs['tooltip'] = tooltip
            res.append('  "%s" [%s];\n' % (name, self._format_node_attrs(this_node_attrs)))
            for base_name in bases:
                res.append('  "%s" -> "%s" [%s];\n' % (base_name, name, self._format_node_attrs(e_attrs)))
        res.append('}\n')
        return ''.join(res)

class inheritance_diagram(graphviz):
    """
    A docutils node to use as a placeholder for the inheritance diagram.
    """
    pass

class InheritanceDiagram(SphinxDirective):
    """
    Run when the inheritance_diagram directive is first encountered.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {'parts': int, 'private-bases': directives.flag, 'caption': directives.unchanged, 'top-classes': directives.unchanged_required}

    def run(self) -> list[Node]:
        if False:
            print('Hello World!')
        node = inheritance_diagram()
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
            graph = InheritanceGraph(class_names, self.env.ref_context.get('py:module'), parts=node['parts'], private_bases='private-bases' in self.options, aliases=self.config.inheritance_alias, top_classes=node['top-classes'])
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

def get_graph_hash(node: inheritance_diagram) -> str:
    if False:
        for i in range(10):
            print('nop')
    encoded = (node['content'] + str(node['parts'])).encode()
    return hashlib.md5(encoded, usedforsecurity=False).hexdigest()[-10:]

def html_visit_inheritance_diagram(self: HTML5Translator, node: inheritance_diagram) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Output the graph for HTML.  This will insert a PNG with clickable\n    image map.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    graphviz_output_format = self.builder.env.config.graphviz_output_format.upper()
    current_filename = path.basename(self.builder.current_docname + self.builder.out_suffix)
    urls = {}
    pending_xrefs = cast(Iterable[addnodes.pending_xref], node)
    for child in pending_xrefs:
        if child.get('refuri') is not None:
            if not child.get('internal', True):
                refname = child['refuri'].rsplit('#', 1)[-1]
            else:
                refname = child['reftitle']
            urls[refname] = child.get('refuri')
        elif child.get('refid') is not None:
            if graphviz_output_format == 'SVG':
                urls[child['reftitle']] = current_filename + '#' + child.get('refid')
            else:
                urls[child['reftitle']] = '#' + child.get('refid')
    dotcode = graph.generate_dot(name, urls, env=self.builder.env)
    render_dot_html(self, node, dotcode, {}, 'inheritance', 'inheritance', alt='Inheritance diagram of ' + node['content'])
    raise nodes.SkipNode

def latex_visit_inheritance_diagram(self: LaTeXTranslator, node: inheritance_diagram) -> None:
    if False:
        while True:
            i = 10
    '\n    Output the graph for LaTeX.  This will insert a PDF.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    dotcode = graph.generate_dot(name, env=self.builder.env, graph_attrs={'size': '"6.0,6.0"'})
    render_dot_latex(self, node, dotcode, {}, 'inheritance')
    raise nodes.SkipNode

def texinfo_visit_inheritance_diagram(self: TexinfoTranslator, node: inheritance_diagram) -> None:
    if False:
        print('Hello World!')
    '\n    Output the graph for Texinfo.  This will insert a PNG.\n    '
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    dotcode = graph.generate_dot(name, env=self.builder.env, graph_attrs={'size': '"6.0,6.0"'})
    render_dot_texinfo(self, node, dotcode, {}, 'inheritance')
    raise nodes.SkipNode

def skip(self: nodes.NodeVisitor, node: inheritance_diagram) -> None:
    if False:
        for i in range(10):
            print('nop')
    raise nodes.SkipNode

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        print('Hello World!')
    app.setup_extension('sphinx.ext.graphviz')
    app.add_node(inheritance_diagram, latex=(latex_visit_inheritance_diagram, None), html=(html_visit_inheritance_diagram, None), text=(skip, None), man=(skip, None), texinfo=(texinfo_visit_inheritance_diagram, None))
    app.add_directive('inheritance-diagram', InheritanceDiagram)
    app.add_config_value('inheritance_graph_attrs', {}, False)
    app.add_config_value('inheritance_node_attrs', {}, False)
    app.add_config_value('inheritance_edge_attrs', {}, False)
    app.add_config_value('inheritance_alias', {}, False)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}