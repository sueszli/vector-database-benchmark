"""
Read graphs in GML format.

"GML, the Graph Modelling Language, is our proposal for a portable
file format for graphs. GML's key features are portability, simple
syntax, extensibility and flexibility. A GML file consists of a
hierarchical key-value lists. Graphs can be annotated with arbitrary
data structures. The idea for a common file format was born at the
GD'95; this proposal is the outcome of many discussions. GML is the
standard file format in the Graphlet graph editor system. It has been
overtaken and adapted by several other systems for drawing graphs."

GML files are stored using a 7-bit ASCII encoding with any extended
ASCII characters (iso8859-1) appearing as HTML character entities.
You will need to give some thought into how the exported data should
interact with different languages and even different Python versions.
Re-importing from gml is also a concern.

Without specifying a `stringizer`/`destringizer`, the code is capable of
writing `int`/`float`/`str`/`dict`/`list` data as required by the GML 
specification.  For writing other data types, and for reading data other
than `str` you need to explicitly supply a `stringizer`/`destringizer`.

For additional documentation on the GML file format, please see the
`GML website <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.

Several example graphs in GML format may be found on Mark Newman's
`Network data page <http://www-personal.umich.edu/~mejn/netdata/>`_.
"""
import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
__all__ = ['read_gml', 'parse_gml', 'generate_gml', 'write_gml']

def escape(text):
    if False:
        return 10
    'Use XML character references to escape characters.\n\n    Use XML character references for unprintable or non-ASCII\n    characters, double quotes and ampersands in a string\n    '

    def fixup(m):
        if False:
            for i in range(10):
                print('nop')
        ch = m.group(0)
        return '&#' + str(ord(ch)) + ';'
    text = re.sub('[^ -~]|[&"]', fixup, text)
    return text if isinstance(text, str) else str(text)

def unescape(text):
    if False:
        return 10
    'Replace XML character references with the referenced characters'

    def fixup(m):
        if False:
            for i in range(10):
                print('nop')
        text = m.group(0)
        if text[1] == '#':
            if text[2] == 'x':
                code = int(text[3:-1], 16)
            else:
                code = int(text[2:-1])
        else:
            try:
                code = htmlentitydefs.name2codepoint[text[1:-1]]
            except KeyError:
                return text
        try:
            return chr(code)
        except (ValueError, OverflowError):
            return text
    return re.sub('&(?:[0-9A-Za-z]+|#(?:[0-9]+|x[0-9A-Fa-f]+));', fixup, text)

def literal_destringizer(rep):
    if False:
        print('Hello World!')
    'Convert a Python literal to the value it represents.\n\n    Parameters\n    ----------\n    rep : string\n        A Python literal.\n\n    Returns\n    -------\n    value : object\n        The value of the Python literal.\n\n    Raises\n    ------\n    ValueError\n        If `rep` is not a Python literal.\n    '
    if isinstance(rep, str):
        orig_rep = rep
        try:
            return literal_eval(rep)
        except SyntaxError as err:
            raise ValueError(f'{orig_rep!r} is not a valid Python literal') from err
    else:
        raise ValueError(f'{rep!r} is not a string')

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_gml(path, label='label', destringizer=None):
    if False:
        i = 10
        return i + 15
    'Read graph in GML format from `path`.\n\n    Parameters\n    ----------\n    path : filename or filehandle\n        The filename or filehandle to read from.\n\n    label : string, optional\n        If not None, the parsed nodes will be renamed according to node\n        attributes indicated by `label`. Default value: \'label\'.\n\n    destringizer : callable, optional\n        A `destringizer` that recovers values stored as strings in GML. If it\n        cannot convert a string to a value, a `ValueError` is raised. Default\n        value : None.\n\n    Returns\n    -------\n    G : NetworkX graph\n        The parsed graph.\n\n    Raises\n    ------\n    NetworkXError\n        If the input cannot be parsed.\n\n    See Also\n    --------\n    write_gml, parse_gml\n    literal_destringizer\n\n    Notes\n    -----\n    GML files are stored using a 7-bit ASCII encoding with any extended\n    ASCII characters (iso8859-1) appearing as HTML character entities.\n    Without specifying a `stringizer`/`destringizer`, the code is capable of\n    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML\n    specification.  For writing other data types, and for reading data other\n    than `str` you need to explicitly supply a `stringizer`/`destringizer`.\n\n    For additional documentation on the GML file format, please see the\n    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.\n\n    See the module docstring :mod:`networkx.readwrite.gml` for more details.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_gml(G, "test.gml")\n\n    GML values are interpreted as strings by default:\n\n    >>> H = nx.read_gml("test.gml")\n    >>> H.nodes\n    NodeView((\'0\', \'1\', \'2\', \'3\'))\n\n    When a `destringizer` is provided, GML values are converted to the provided type.\n    For example, integer nodes can be recovered as shown below:\n\n    >>> J = nx.read_gml("test.gml", destringizer=int)\n    >>> J.nodes\n    NodeView((0, 1, 2, 3))\n\n    '

    def filter_lines(lines):
        if False:
            print('Hello World!')
        for line in lines:
            try:
                line = line.decode('ascii')
            except UnicodeDecodeError as err:
                raise NetworkXError('input is not ASCII-encoded') from err
            if not isinstance(line, str):
                lines = str(lines)
            if line and line[-1] == '\n':
                line = line[:-1]
            yield line
    G = parse_gml_lines(filter_lines(path), label, destringizer)
    return G

@nx._dispatch(graphs=None)
def parse_gml(lines, label='label', destringizer=None):
    if False:
        while True:
            i = 10
    "Parse GML graph from a string or iterable.\n\n    Parameters\n    ----------\n    lines : string or iterable of strings\n       Data in GML format.\n\n    label : string, optional\n        If not None, the parsed nodes will be renamed according to node\n        attributes indicated by `label`. Default value: 'label'.\n\n    destringizer : callable, optional\n        A `destringizer` that recovers values stored as strings in GML. If it\n        cannot convert a string to a value, a `ValueError` is raised. Default\n        value : None.\n\n    Returns\n    -------\n    G : NetworkX graph\n        The parsed graph.\n\n    Raises\n    ------\n    NetworkXError\n        If the input cannot be parsed.\n\n    See Also\n    --------\n    write_gml, read_gml\n\n    Notes\n    -----\n    This stores nested GML attributes as dictionaries in the NetworkX graph,\n    node, and edge attribute structures.\n\n    GML files are stored using a 7-bit ASCII encoding with any extended\n    ASCII characters (iso8859-1) appearing as HTML character entities.\n    Without specifying a `stringizer`/`destringizer`, the code is capable of\n    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML\n    specification.  For writing other data types, and for reading data other\n    than `str` you need to explicitly supply a `stringizer`/`destringizer`.\n\n    For additional documentation on the GML file format, please see the\n    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.\n\n    See the module docstring :mod:`networkx.readwrite.gml` for more details.\n    "

    def decode_line(line):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(line, bytes):
            try:
                line.decode('ascii')
            except UnicodeDecodeError as err:
                raise NetworkXError('input is not ASCII-encoded') from err
        if not isinstance(line, str):
            line = str(line)
        return line

    def filter_lines(lines):
        if False:
            return 10
        if isinstance(lines, str):
            lines = decode_line(lines)
            lines = lines.splitlines()
            yield from lines
        else:
            for line in lines:
                line = decode_line(line)
                if line and line[-1] == '\n':
                    line = line[:-1]
                if line.find('\n') != -1:
                    raise NetworkXError('input line contains newline')
                yield line
    G = parse_gml_lines(filter_lines(lines), label, destringizer)
    return G

class Pattern(Enum):
    """encodes the index of each token-matching pattern in `tokenize`."""
    KEYS = 0
    REALS = 1
    INTS = 2
    STRINGS = 3
    DICT_START = 4
    DICT_END = 5
    COMMENT_WHITESPACE = 6

class Token(NamedTuple):
    category: Pattern
    value: Any
    line: int
    position: int
LIST_START_VALUE = '_networkx_list_start'

def parse_gml_lines(lines, label, destringizer):
    if False:
        i = 10
        return i + 15
    'Parse GML `lines` into a graph.'

    def tokenize():
        if False:
            i = 10
            return i + 15
        patterns = ['[A-Za-z][0-9A-Za-z_]*\\b', '[+-]?(?:[0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*|INF)(?:[Ee][+-]?[0-9]+)?', '[+-]?[0-9]+', '".*?"', '\\[', '\\]', '#.*$|\\s+']
        tokens = re.compile('|'.join((f'({pattern})' for pattern in patterns)))
        lineno = 0
        multilines = []
        for line in lines:
            pos = 0
            if multilines:
                multilines.append(line.strip())
                if line[-1] == '"':
                    line = ' '.join(multilines)
                    multilines = []
                else:
                    lineno += 1
                    continue
            elif line.count('"') == 1:
                if line.strip()[0] != '"' and line.strip()[-1] != '"':
                    multilines = [line.rstrip()]
                    lineno += 1
                    continue
            length = len(line)
            while pos < length:
                match = tokens.match(line, pos)
                if match is None:
                    m = f'cannot tokenize {line[pos:]} at ({lineno + 1}, {pos + 1})'
                    raise NetworkXError(m)
                for i in range(len(patterns)):
                    group = match.group(i + 1)
                    if group is not None:
                        if i == 0:
                            value = group.rstrip()
                        elif i == 1:
                            value = float(group)
                        elif i == 2:
                            value = int(group)
                        else:
                            value = group
                        if i != 6:
                            yield Token(Pattern(i), value, lineno + 1, pos + 1)
                        pos += len(group)
                        break
            lineno += 1
        yield Token(None, None, lineno + 1, 1)

    def unexpected(curr_token, expected):
        if False:
            while True:
                i = 10
        (category, value, lineno, pos) = curr_token
        value = repr(value) if value is not None else 'EOF'
        raise NetworkXError(f'expected {expected}, found {value} at ({lineno}, {pos})')

    def consume(curr_token, category, expected):
        if False:
            print('Hello World!')
        if curr_token.category == category:
            return next(tokens)
        unexpected(curr_token, expected)

    def parse_kv(curr_token):
        if False:
            for i in range(10):
                print('nop')
        dct = defaultdict(list)
        while curr_token.category == Pattern.KEYS:
            key = curr_token.value
            curr_token = next(tokens)
            category = curr_token.category
            if category == Pattern.REALS or category == Pattern.INTS:
                value = curr_token.value
                curr_token = next(tokens)
            elif category == Pattern.STRINGS:
                value = unescape(curr_token.value[1:-1])
                if destringizer:
                    try:
                        value = destringizer(value)
                    except ValueError:
                        pass
                if value == '()':
                    value = ()
                if value == '[]':
                    value = []
                curr_token = next(tokens)
            elif category == Pattern.DICT_START:
                (curr_token, value) = parse_dict(curr_token)
            elif key in ('id', 'label', 'source', 'target'):
                try:
                    value = unescape(str(curr_token.value))
                    if destringizer:
                        try:
                            value = destringizer(value)
                        except ValueError:
                            pass
                    curr_token = next(tokens)
                except Exception:
                    msg = "an int, float, string, '[' or string" + ' convertible ASCII value for node id or label'
                    unexpected(curr_token, msg)
            elif curr_token.value in {'NAN', 'INF'}:
                value = float(curr_token.value)
                curr_token = next(tokens)
            else:
                unexpected(curr_token, "an int, float, string or '['")
            dct[key].append(value)

        def clean_dict_value(value):
            if False:
                print('Hello World!')
            if not isinstance(value, list):
                return value
            if len(value) == 1:
                return value[0]
            if value[0] == LIST_START_VALUE:
                return value[1:]
            return value
        dct = {key: clean_dict_value(value) for (key, value) in dct.items()}
        return (curr_token, dct)

    def parse_dict(curr_token):
        if False:
            for i in range(10):
                print('nop')
        curr_token = consume(curr_token, Pattern.DICT_START, "'['")
        (curr_token, dct) = parse_kv(curr_token)
        curr_token = consume(curr_token, Pattern.DICT_END, "']'")
        return (curr_token, dct)

    def parse_graph():
        if False:
            for i in range(10):
                print('nop')
        (curr_token, dct) = parse_kv(next(tokens))
        if curr_token.category is not None:
            unexpected(curr_token, 'EOF')
        if 'graph' not in dct:
            raise NetworkXError('input contains no graph')
        graph = dct['graph']
        if isinstance(graph, list):
            raise NetworkXError('input contains more than one graph')
        return graph
    tokens = tokenize()
    graph = parse_graph()
    directed = graph.pop('directed', False)
    multigraph = graph.pop('multigraph', False)
    if not multigraph:
        G = nx.DiGraph() if directed else nx.Graph()
    else:
        G = nx.MultiDiGraph() if directed else nx.MultiGraph()
    graph_attr = {k: v for (k, v) in graph.items() if k not in ('node', 'edge')}
    G.graph.update(graph_attr)

    def pop_attr(dct, category, attr, i):
        if False:
            while True:
                i = 10
        try:
            return dct.pop(attr)
        except KeyError as err:
            raise NetworkXError(f'{category} #{i} has no {attr!r} attribute') from err
    nodes = graph.get('node', [])
    mapping = {}
    node_labels = set()
    for (i, node) in enumerate(nodes if isinstance(nodes, list) else [nodes]):
        id = pop_attr(node, 'node', 'id', i)
        if id in G:
            raise NetworkXError(f'node id {id!r} is duplicated')
        if label is not None and label != 'id':
            node_label = pop_attr(node, 'node', label, i)
            if node_label in node_labels:
                raise NetworkXError(f'node label {node_label!r} is duplicated')
            node_labels.add(node_label)
            mapping[id] = node_label
        G.add_node(id, **node)
    edges = graph.get('edge', [])
    for (i, edge) in enumerate(edges if isinstance(edges, list) else [edges]):
        source = pop_attr(edge, 'edge', 'source', i)
        target = pop_attr(edge, 'edge', 'target', i)
        if source not in G:
            raise NetworkXError(f'edge #{i} has undefined source {source!r}')
        if target not in G:
            raise NetworkXError(f'edge #{i} has undefined target {target!r}')
        if not multigraph:
            if not G.has_edge(source, target):
                G.add_edge(source, target, **edge)
            else:
                arrow = '->' if directed else '--'
                msg = f'edge #{i} ({source!r}{arrow}{target!r}) is duplicated'
                raise nx.NetworkXError(msg)
        else:
            key = edge.pop('key', None)
            if key is not None and G.has_edge(source, target, key):
                arrow = '->' if directed else '--'
                msg = f'edge #{i} ({source!r}{arrow}{target!r}, {key!r})'
                msg2 = 'Hint: If multigraph add "multigraph 1" to file header.'
                raise nx.NetworkXError(msg + ' is duplicated\n' + msg2)
            G.add_edge(source, target, key, **edge)
    if label is not None and label != 'id':
        G = nx.relabel_nodes(G, mapping)
    return G

def literal_stringizer(value):
    if False:
        while True:
            i = 10
    'Convert a `value` to a Python literal in GML representation.\n\n    Parameters\n    ----------\n    value : object\n        The `value` to be converted to GML representation.\n\n    Returns\n    -------\n    rep : string\n        A double-quoted Python literal representing value. Unprintable\n        characters are replaced by XML character references.\n\n    Raises\n    ------\n    ValueError\n        If `value` cannot be converted to GML.\n\n    Notes\n    -----\n    The original value can be recovered using the\n    :func:`networkx.readwrite.gml.literal_destringizer` function.\n    '

    def stringize(value):
        if False:
            return 10
        if isinstance(value, int | bool) or value is None:
            if value is True:
                buf.write(str(1))
            elif value is False:
                buf.write(str(0))
            else:
                buf.write(str(value))
        elif isinstance(value, str):
            text = repr(value)
            if text[0] != 'u':
                try:
                    value.encode('latin1')
                except UnicodeEncodeError:
                    text = 'u' + text
            buf.write(text)
        elif isinstance(value, float | complex | str | bytes):
            buf.write(repr(value))
        elif isinstance(value, list):
            buf.write('[')
            first = True
            for item in value:
                if not first:
                    buf.write(',')
                else:
                    first = False
                stringize(item)
            buf.write(']')
        elif isinstance(value, tuple):
            if len(value) > 1:
                buf.write('(')
                first = True
                for item in value:
                    if not first:
                        buf.write(',')
                    else:
                        first = False
                    stringize(item)
                buf.write(')')
            elif value:
                buf.write('(')
                stringize(value[0])
                buf.write(',)')
            else:
                buf.write('()')
        elif isinstance(value, dict):
            buf.write('{')
            first = True
            for (key, value) in value.items():
                if not first:
                    buf.write(',')
                else:
                    first = False
                stringize(key)
                buf.write(':')
                stringize(value)
            buf.write('}')
        elif isinstance(value, set):
            buf.write('{')
            first = True
            for item in value:
                if not first:
                    buf.write(',')
                else:
                    first = False
                stringize(item)
            buf.write('}')
        else:
            msg = f'{value!r} cannot be converted into a Python literal'
            raise ValueError(msg)
    buf = StringIO()
    stringize(value)
    return buf.getvalue()

def generate_gml(G, stringizer=None):
    if False:
        return 10
    'Generate a single entry of the graph `G` in GML format.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The graph to be converted to GML.\n\n    stringizer : callable, optional\n        A `stringizer` which converts non-int/non-float/non-dict values into\n        strings. If it cannot convert a value into a string, it should raise a\n        `ValueError` to indicate that. Default value: None.\n\n    Returns\n    -------\n    lines: generator of strings\n        Lines of GML data. Newlines are not appended.\n\n    Raises\n    ------\n    NetworkXError\n        If `stringizer` cannot convert a value into a string, or the value to\n        convert is not a string while `stringizer` is None.\n\n    See Also\n    --------\n    literal_stringizer\n\n    Notes\n    -----\n    Graph attributes named \'directed\', \'multigraph\', \'node\' or\n    \'edge\', node attributes named \'id\' or \'label\', edge attributes\n    named \'source\' or \'target\' (or \'key\' if `G` is a multigraph)\n    are ignored because these attribute names are used to encode the graph\n    structure.\n\n    GML files are stored using a 7-bit ASCII encoding with any extended\n    ASCII characters (iso8859-1) appearing as HTML character entities.\n    Without specifying a `stringizer`/`destringizer`, the code is capable of\n    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML\n    specification.  For writing other data types, and for reading data other\n    than `str` you need to explicitly supply a `stringizer`/`destringizer`.\n\n    For additional documentation on the GML file format, please see the\n    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.\n\n    See the module docstring :mod:`networkx.readwrite.gml` for more details.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_node("1")\n    >>> print("\\n".join(nx.generate_gml(G)))\n    graph [\n      node [\n        id 0\n        label "1"\n      ]\n    ]\n    >>> G = nx.MultiGraph([("a", "b"), ("a", "b")])\n    >>> print("\\n".join(nx.generate_gml(G)))\n    graph [\n      multigraph 1\n      node [\n        id 0\n        label "a"\n      ]\n      node [\n        id 1\n        label "b"\n      ]\n      edge [\n        source 0\n        target 1\n        key 0\n      ]\n      edge [\n        source 0\n        target 1\n        key 1\n      ]\n    ]\n    '
    valid_keys = re.compile('^[A-Za-z][0-9A-Za-z_]*$')

    def stringize(key, value, ignored_keys, indent, in_list=False):
        if False:
            while True:
                i = 10
        if not isinstance(key, str):
            raise NetworkXError(f'{key!r} is not a string')
        if not valid_keys.match(key):
            raise NetworkXError(f'{key!r} is not a valid key')
        if not isinstance(key, str):
            key = str(key)
        if key not in ignored_keys:
            if isinstance(value, int | bool):
                if key == 'label':
                    yield (indent + key + ' "' + str(value) + '"')
                elif value is True:
                    yield (indent + key + ' 1')
                elif value is False:
                    yield (indent + key + ' 0')
                elif value < -2 ** 31 or value >= 2 ** 31:
                    yield (indent + key + ' "' + str(value) + '"')
                else:
                    yield (indent + key + ' ' + str(value))
            elif isinstance(value, float):
                text = repr(value).upper()
                if text == repr(float('inf')).upper():
                    text = '+' + text
                else:
                    epos = text.rfind('E')
                    if epos != -1 and text.find('.', 0, epos) == -1:
                        text = text[:epos] + '.' + text[epos:]
                if key == 'label':
                    yield (indent + key + ' "' + text + '"')
                else:
                    yield (indent + key + ' ' + text)
            elif isinstance(value, dict):
                yield (indent + key + ' [')
                next_indent = indent + '  '
                for (key, value) in value.items():
                    yield from stringize(key, value, (), next_indent)
                yield (indent + ']')
            elif isinstance(value, tuple) and key == 'label':
                yield (indent + key + f''' "({','.join((repr(v) for v in value))})"''')
            elif isinstance(value, list | tuple) and key != 'label' and (not in_list):
                if len(value) == 0:
                    yield (indent + key + ' ' + f'"{value!r}"')
                if len(value) == 1:
                    yield (indent + key + ' ' + f'"{LIST_START_VALUE}"')
                for val in value:
                    yield from stringize(key, val, (), indent, True)
            else:
                if stringizer:
                    try:
                        value = stringizer(value)
                    except ValueError as err:
                        raise NetworkXError(f'{value!r} cannot be converted into a string') from err
                if not isinstance(value, str):
                    raise NetworkXError(f'{value!r} is not a string')
                yield (indent + key + ' "' + escape(value) + '"')
    multigraph = G.is_multigraph()
    yield 'graph ['
    if G.is_directed():
        yield '  directed 1'
    if multigraph:
        yield '  multigraph 1'
    ignored_keys = {'directed', 'multigraph', 'node', 'edge'}
    for (attr, value) in G.graph.items():
        yield from stringize(attr, value, ignored_keys, '  ')
    node_id = dict(zip(G, range(len(G))))
    ignored_keys = {'id', 'label'}
    for (node, attrs) in G.nodes.items():
        yield '  node ['
        yield ('    id ' + str(node_id[node]))
        yield from stringize('label', node, (), '    ')
        for (attr, value) in attrs.items():
            yield from stringize(attr, value, ignored_keys, '    ')
        yield '  ]'
    ignored_keys = {'source', 'target'}
    kwargs = {'data': True}
    if multigraph:
        ignored_keys.add('key')
        kwargs['keys'] = True
    for e in G.edges(**kwargs):
        yield '  edge ['
        yield ('    source ' + str(node_id[e[0]]))
        yield ('    target ' + str(node_id[e[1]]))
        if multigraph:
            yield from stringize('key', e[2], (), '    ')
        for (attr, value) in e[-1].items():
            yield from stringize(attr, value, ignored_keys, '    ')
        yield '  ]'
    yield ']'

@open_file(1, mode='wb')
def write_gml(G, path, stringizer=None):
    if False:
        for i in range(10):
            print('nop')
    'Write a graph `G` in GML format to the file or file handle `path`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The graph to be converted to GML.\n\n    path : filename or filehandle\n        The filename or filehandle to write. Files whose names end with .gz or\n        .bz2 will be compressed.\n\n    stringizer : callable, optional\n        A `stringizer` which converts non-int/non-float/non-dict values into\n        strings. If it cannot convert a value into a string, it should raise a\n        `ValueError` to indicate that. Default value: None.\n\n    Raises\n    ------\n    NetworkXError\n        If `stringizer` cannot convert a value into a string, or the value to\n        convert is not a string while `stringizer` is None.\n\n    See Also\n    --------\n    read_gml, generate_gml\n    literal_stringizer\n\n    Notes\n    -----\n    Graph attributes named \'directed\', \'multigraph\', \'node\' or\n    \'edge\', node attributes named \'id\' or \'label\', edge attributes\n    named \'source\' or \'target\' (or \'key\' if `G` is a multigraph)\n    are ignored because these attribute names are used to encode the graph\n    structure.\n\n    GML files are stored using a 7-bit ASCII encoding with any extended\n    ASCII characters (iso8859-1) appearing as HTML character entities.\n    Without specifying a `stringizer`/`destringizer`, the code is capable of\n    writing `int`/`float`/`str`/`dict`/`list` data as required by the GML\n    specification.  For writing other data types, and for reading data other\n    than `str` you need to explicitly supply a `stringizer`/`destringizer`.\n\n    Note that while we allow non-standard GML to be read from a file, we make\n    sure to write GML format. In particular, underscores are not allowed in\n    attribute names.\n    For additional documentation on the GML file format, please see the\n    `GML url <https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1>`_.\n\n    See the module docstring :mod:`networkx.readwrite.gml` for more details.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_gml(G, "test.gml")\n\n    Filenames ending in .gz or .bz2 will be compressed.\n\n    >>> nx.write_gml(G, "test.gml.gz")\n    '
    for line in generate_gml(G, stringizer):
        path.write((line + '\n').encode('ascii'))