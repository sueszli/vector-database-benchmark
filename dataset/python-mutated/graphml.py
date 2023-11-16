"""
*******
GraphML
*******
Read and write graphs in GraphML format.

.. warning::

    This parser uses the standard xml library present in Python, which is
    insecure - see :external+python:mod:`xml` for additional information.
    Only parse GraphML files you trust.

This implementation does not support mixed graphs (directed and unidirected
edges together), hyperedges, nested graphs, or ports.

"GraphML is a comprehensive and easy-to-use file format for graphs. It
consists of a language core to describe the structural properties of a
graph and a flexible extension mechanism to add application-specific
data. Its main features include support of

    * directed, undirected, and mixed graphs,
    * hypergraphs,
    * hierarchical graphs,
    * graphical representations,
    * references to external data,
    * application-specific attribute data, and
    * light-weight parsers.

Unlike many other file formats for graphs, GraphML does not use a
custom syntax. Instead, it is based on XML and hence ideally suited as
a common denominator for all kinds of services generating, archiving,
or processing graphs."

http://graphml.graphdrawing.org/

Format
------
GraphML is an XML format.  See
http://graphml.graphdrawing.org/specification.html for the specification and
http://graphml.graphdrawing.org/primer/graphml-primer.html
for examples.
"""
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
__all__ = ['write_graphml', 'read_graphml', 'generate_graphml', 'write_graphml_xml', 'write_graphml_lxml', 'parse_graphml', 'GraphMLWriter', 'GraphMLReader']

@open_file(1, mode='wb')
def write_graphml_xml(G, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None):
    if False:
        for i in range(10):
            print('nop')
    'Write G in GraphML XML format to path\n\n    Parameters\n    ----------\n    G : graph\n       A networkx graph\n    path : file or string\n       File or filename to write.\n       Filenames ending in .gz or .bz2 will be compressed.\n    encoding : string (optional)\n       Encoding for text data.\n    prettyprint : bool (optional)\n       If True use line breaks and indenting in output XML.\n    infer_numeric_types : boolean\n       Determine if numeric types should be generalized.\n       For example, if edges have both int and float \'weight\' attributes,\n       we infer in GraphML that both are floats.\n    named_key_ids : bool (optional)\n       If True use attr.name as value for key elements\' id attribute.\n    edge_id_from_attribute : dict key (optional)\n        If provided, the graphml edge id is set by looking up the corresponding\n        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,\n        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_graphml(G, "test.graphml")\n\n    Notes\n    -----\n    This implementation does not support mixed graphs (directed\n    and unidirected edges together) hyperedges, nested graphs, or ports.\n    '
    writer = GraphMLWriter(encoding=encoding, prettyprint=prettyprint, infer_numeric_types=infer_numeric_types, named_key_ids=named_key_ids, edge_id_from_attribute=edge_id_from_attribute)
    writer.add_graph_element(G)
    writer.dump(path)

@open_file(1, mode='wb')
def write_graphml_lxml(G, path, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None):
    if False:
        for i in range(10):
            print('nop')
    'Write G in GraphML XML format to path\n\n    This function uses the LXML framework and should be faster than\n    the version using the xml library.\n\n    Parameters\n    ----------\n    G : graph\n       A networkx graph\n    path : file or string\n       File or filename to write.\n       Filenames ending in .gz or .bz2 will be compressed.\n    encoding : string (optional)\n       Encoding for text data.\n    prettyprint : bool (optional)\n       If True use line breaks and indenting in output XML.\n    infer_numeric_types : boolean\n       Determine if numeric types should be generalized.\n       For example, if edges have both int and float \'weight\' attributes,\n       we infer in GraphML that both are floats.\n    named_key_ids : bool (optional)\n       If True use attr.name as value for key elements\' id attribute.\n    edge_id_from_attribute : dict key (optional)\n        If provided, the graphml edge id is set by looking up the corresponding\n        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,\n        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_graphml_lxml(G, "fourpath.graphml")\n\n    Notes\n    -----\n    This implementation does not support mixed graphs (directed\n    and unidirected edges together) hyperedges, nested graphs, or ports.\n    '
    try:
        import lxml.etree as lxmletree
    except ImportError:
        return write_graphml_xml(G, path, encoding, prettyprint, infer_numeric_types, named_key_ids, edge_id_from_attribute)
    writer = GraphMLWriterLxml(path, graph=G, encoding=encoding, prettyprint=prettyprint, infer_numeric_types=infer_numeric_types, named_key_ids=named_key_ids, edge_id_from_attribute=edge_id_from_attribute)
    writer.dump()

def generate_graphml(G, encoding='utf-8', prettyprint=True, named_key_ids=False, edge_id_from_attribute=None):
    if False:
        for i in range(10):
            print('nop')
    "Generate GraphML lines for G\n\n    Parameters\n    ----------\n    G : graph\n       A networkx graph\n    encoding : string (optional)\n       Encoding for text data.\n    prettyprint : bool (optional)\n       If True use line breaks and indenting in output XML.\n    named_key_ids : bool (optional)\n       If True use attr.name as value for key elements' id attribute.\n    edge_id_from_attribute : dict key (optional)\n        If provided, the graphml edge id is set by looking up the corresponding\n        edge data attribute keyed by this parameter. If `None` or the key does not exist in edge data,\n        the edge id is set by the edge key if `G` is a MultiGraph, else the edge id is left unset.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> linefeed = chr(10)  # linefeed = \n\n    >>> s = linefeed.join(nx.generate_graphml(G))\n    >>> for line in nx.generate_graphml(G):  # doctest: +SKIP\n    ...     print(line)\n\n    Notes\n    -----\n    This implementation does not support mixed graphs (directed and unidirected\n    edges together) hyperedges, nested graphs, or ports.\n    "
    writer = GraphMLWriter(encoding=encoding, prettyprint=prettyprint, named_key_ids=named_key_ids, edge_id_from_attribute=edge_id_from_attribute)
    writer.add_graph_element(G)
    yield from str(writer).splitlines()

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_graphml(path, node_type=str, edge_key_type=int, force_multigraph=False):
    if False:
        print('Hello World!')
    'Read graph in GraphML format from path.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to write.\n       Filenames ending in .gz or .bz2 will be compressed.\n\n    node_type: Python type (default: str)\n       Convert node ids to this type\n\n    edge_key_type: Python type (default: int)\n       Convert graphml edge ids to this type. Multigraphs use id as edge key.\n       Non-multigraphs add to edge attribute dict with name "id".\n\n    force_multigraph : bool (default: False)\n       If True, return a multigraph with edge keys. If False (the default)\n       return a multigraph when multiedges are in the graph.\n\n    Returns\n    -------\n    graph: NetworkX graph\n        If parallel edges are present or `force_multigraph=True` then\n        a MultiGraph or MultiDiGraph is returned. Otherwise a Graph/DiGraph.\n        The returned graph is directed if the file indicates it should be.\n\n    Notes\n    -----\n    Default node and edge attributes are not propagated to each node and edge.\n    They can be obtained from `G.graph` and applied to node and edge attributes\n    if desired using something like this:\n\n    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP\n    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP\n    ...     if "color" not in data:\n    ...         data["color"] = default_color\n    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP\n    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP\n    ...     if "color" not in data:\n    ...         data["color"] = default_color\n\n    This implementation does not support mixed graphs (directed and unidirected\n    edges together), hypergraphs, nested graphs, or ports.\n\n    For multigraphs the GraphML edge "id" will be used as the edge\n    key.  If not specified then they "key" attribute will be used.  If\n    there is no "key" attribute a default NetworkX multigraph edge key\n    will be provided.\n\n    Files with the yEd "yfiles" extension can be read. The type of the node\'s\n    shape is preserved in the `shape_type` node attribute.\n\n    yEd compressed files ("file.graphmlz" extension) can be read by renaming\n    the file to "file.graphml.gz".\n\n    '
    reader = GraphMLReader(node_type, edge_key_type, force_multigraph)
    glist = list(reader(path=path))
    if len(glist) == 0:
        header = b'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        path.seek(0)
        old_bytes = path.read()
        new_bytes = old_bytes.replace(b'<graphml>', header)
        glist = list(reader(string=new_bytes))
        if len(glist) == 0:
            raise nx.NetworkXError('file not successfully read as graphml')
    return glist[0]

@nx._dispatch(graphs=None)
def parse_graphml(graphml_string, node_type=str, edge_key_type=int, force_multigraph=False):
    if False:
        return 10
    'Read graph in GraphML format from string.\n\n    Parameters\n    ----------\n    graphml_string : string\n       String containing graphml information\n       (e.g., contents of a graphml file).\n\n    node_type: Python type (default: str)\n       Convert node ids to this type\n\n    edge_key_type: Python type (default: int)\n       Convert graphml edge ids to this type. Multigraphs use id as edge key.\n       Non-multigraphs add to edge attribute dict with name "id".\n\n    force_multigraph : bool (default: False)\n       If True, return a multigraph with edge keys. If False (the default)\n       return a multigraph when multiedges are in the graph.\n\n\n    Returns\n    -------\n    graph: NetworkX graph\n        If no parallel edges are found a Graph or DiGraph is returned.\n        Otherwise a MultiGraph or MultiDiGraph is returned.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> linefeed = chr(10)  # linefeed = \n\n    >>> s = linefeed.join(nx.generate_graphml(G))\n    >>> H = nx.parse_graphml(s)\n\n    Notes\n    -----\n    Default node and edge attributes are not propagated to each node and edge.\n    They can be obtained from `G.graph` and applied to node and edge attributes\n    if desired using something like this:\n\n    >>> default_color = G.graph["node_default"]["color"]  # doctest: +SKIP\n    >>> for node, data in G.nodes(data=True):  # doctest: +SKIP\n    ...     if "color" not in data:\n    ...         data["color"] = default_color\n    >>> default_color = G.graph["edge_default"]["color"]  # doctest: +SKIP\n    >>> for u, v, data in G.edges(data=True):  # doctest: +SKIP\n    ...     if "color" not in data:\n    ...         data["color"] = default_color\n\n    This implementation does not support mixed graphs (directed and unidirected\n    edges together), hypergraphs, nested graphs, or ports.\n\n    For multigraphs the GraphML edge "id" will be used as the edge\n    key.  If not specified then they "key" attribute will be used.  If\n    there is no "key" attribute a default NetworkX multigraph edge key\n    will be provided.\n\n    '
    reader = GraphMLReader(node_type, edge_key_type, force_multigraph)
    glist = list(reader(string=graphml_string))
    if len(glist) == 0:
        header = '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        new_string = graphml_string.replace('<graphml>', header)
        glist = list(reader(string=new_string))
        if len(glist) == 0:
            raise nx.NetworkXError('file not successfully read as graphml')
    return glist[0]

class GraphML:
    NS_GRAPHML = 'http://graphml.graphdrawing.org/xmlns'
    NS_XSI = 'http://www.w3.org/2001/XMLSchema-instance'
    NS_Y = 'http://www.yworks.com/xml/graphml'
    SCHEMALOCATION = ' '.join(['http://graphml.graphdrawing.org/xmlns', 'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'])

    def construct_types(self):
        if False:
            while True:
                i = 10
        types = [(int, 'integer'), (str, 'yfiles'), (str, 'string'), (int, 'int'), (int, 'long'), (float, 'float'), (float, 'double'), (bool, 'boolean')]
        try:
            import numpy as np
        except:
            pass
        else:
            types = [(np.float64, 'float'), (np.float32, 'float'), (np.float16, 'float'), (np.int_, 'int'), (np.int8, 'int'), (np.int16, 'int'), (np.int32, 'int'), (np.int64, 'int'), (np.uint8, 'int'), (np.uint16, 'int'), (np.uint32, 'int'), (np.uint64, 'int'), (np.int_, 'int'), (np.intc, 'int'), (np.intp, 'int')] + types
        self.xml_type = dict(types)
        self.python_type = dict((reversed(a) for a in types))
    convert_bool = {'true': True, 'false': False, '0': False, 0: False, '1': True, 1: True}

    def get_xml_type(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Wrapper around the xml_type dict that raises a more informative\n        exception message when a user attempts to use data of a type not\n        supported by GraphML.'
        try:
            return self.xml_type[key]
        except KeyError as err:
            raise TypeError(f'GraphML does not support type {key} as data values.') from err

class GraphMLWriter(GraphML):

    def __init__(self, graph=None, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None):
        if False:
            i = 10
            return i + 15
        self.construct_types()
        from xml.etree.ElementTree import Element
        self.myElement = Element
        self.infer_numeric_types = infer_numeric_types
        self.prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.encoding = encoding
        self.xml = self.myElement('graphml', {'xmlns': self.NS_GRAPHML, 'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation': self.SCHEMALOCATION})
        self.keys = {}
        self.attributes = defaultdict(list)
        self.attribute_types = defaultdict(set)
        if graph is not None:
            self.add_graph_element(graph)

    def __str__(self):
        if False:
            print('Hello World!')
        from xml.etree.ElementTree import tostring
        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

    def attr_type(self, name, scope, value):
        if False:
            return 10
        "Infer the attribute type of data named name. Currently this only\n        supports inference of numeric types.\n\n        If self.infer_numeric_types is false, type is used. Otherwise, pick the\n        most general of types found across all values with name and scope. This\n        means edges with data named 'weight' are treated separately from nodes\n        with data named 'weight'.\n        "
        if self.infer_numeric_types:
            types = self.attribute_types[name, scope]
            if len(types) > 1:
                types = {self.get_xml_type(t) for t in types}
                if 'string' in types:
                    return str
                elif 'float' in types or 'double' in types:
                    return float
                else:
                    return int
            else:
                return list(types)[0]
        else:
            return type(value)

    def get_key(self, name, attr_type, scope, default):
        if False:
            print('Hello World!')
        keys_key = (name, attr_type, scope)
        try:
            return self.keys[keys_key]
        except KeyError:
            if self.named_key_ids:
                new_id = name
            else:
                new_id = f'd{len(list(self.keys))}'
            self.keys[keys_key] = new_id
            key_kwargs = {'id': new_id, 'for': scope, 'attr.name': name, 'attr.type': attr_type}
            key_element = self.myElement('key', **key_kwargs)
            if default is not None:
                default_element = self.myElement('default')
                default_element.text = str(default)
                key_element.append(default_element)
            self.xml.insert(0, key_element)
        return new_id

    def add_data(self, name, element_type, value, scope='all', default=None):
        if False:
            while True:
                i = 10
        '\n        Make a data element for an edge or a node. Keep a log of the\n        type in the keys table.\n        '
        if element_type not in self.xml_type:
            raise nx.NetworkXError(f'GraphML writer does not support {element_type} as data values.')
        keyid = self.get_key(name, self.get_xml_type(element_type), scope, default)
        data_element = self.myElement('data', key=keyid)
        data_element.text = str(value)
        return data_element

    def add_attributes(self, scope, xml_obj, data, default):
        if False:
            while True:
                i = 10
        'Appends attribute data to edges or nodes, and stores type information\n        to be added later. See add_graph_element.\n        '
        for (k, v) in data.items():
            self.attribute_types[str(k), scope].add(type(v))
            self.attributes[xml_obj].append([k, v, scope, default.get(k)])

    def add_nodes(self, G, graph_element):
        if False:
            return 10
        default = G.graph.get('node_default', {})
        for (node, data) in G.nodes(data=True):
            node_element = self.myElement('node', id=str(node))
            self.add_attributes('node', node_element, data, default)
            graph_element.append(node_element)

    def add_edges(self, G, graph_element):
        if False:
            print('Hello World!')
        if G.is_multigraph():
            for (u, v, key, data) in G.edges(data=True, keys=True):
                edge_element = self.myElement('edge', source=str(u), target=str(v), id=str(data.get(self.edge_id_from_attribute)) if self.edge_id_from_attribute and self.edge_id_from_attribute in data else str(key))
                default = G.graph.get('edge_default', {})
                self.add_attributes('edge', edge_element, data, default)
                graph_element.append(edge_element)
        else:
            for (u, v, data) in G.edges(data=True):
                if self.edge_id_from_attribute and self.edge_id_from_attribute in data:
                    edge_element = self.myElement('edge', source=str(u), target=str(v), id=str(data.get(self.edge_id_from_attribute)))
                else:
                    edge_element = self.myElement('edge', source=str(u), target=str(v))
                default = G.graph.get('edge_default', {})
                self.add_attributes('edge', edge_element, data, default)
                graph_element.append(edge_element)

    def add_graph_element(self, G):
        if False:
            print('Hello World!')
        '\n        Serialize graph G in GraphML to the stream.\n        '
        if G.is_directed():
            default_edge_type = 'directed'
        else:
            default_edge_type = 'undirected'
        graphid = G.graph.pop('id', None)
        if graphid is None:
            graph_element = self.myElement('graph', edgedefault=default_edge_type)
        else:
            graph_element = self.myElement('graph', edgedefault=default_edge_type, id=graphid)
        default = {}
        data = {k: v for (k, v) in G.graph.items() if k not in ['node_default', 'edge_default']}
        self.add_attributes('graph', graph_element, data, default)
        self.add_nodes(G, graph_element)
        self.add_edges(G, graph_element)
        for (xml_obj, data) in self.attributes.items():
            for (k, v, scope, default) in data:
                xml_obj.append(self.add_data(str(k), self.attr_type(k, scope, v), str(v), scope, default))
        self.xml.append(graph_element)

    def add_graphs(self, graph_list):
        if False:
            return 10
        'Add many graphs to this GraphML document.'
        for G in graph_list:
            self.add_graph_element(G)

    def dump(self, stream):
        if False:
            print('Hello World!')
        from xml.etree.ElementTree import ElementTree
        if self.prettyprint:
            self.indent(self.xml)
        document = ElementTree(self.xml)
        document.write(stream, encoding=self.encoding, xml_declaration=True)

    def indent(self, elem, level=0):
        if False:
            print('Hello World!')
        i = '\n' + level * '  '
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + '  '
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

class IncrementalElement:
    """Wrapper for _IncrementalWriter providing an Element like interface.

    This wrapper does not intend to be a complete implementation but rather to
    deal with those calls used in GraphMLWriter.
    """

    def __init__(self, xml, prettyprint):
        if False:
            for i in range(10):
                print('nop')
        self.xml = xml
        self.prettyprint = prettyprint

    def append(self, element):
        if False:
            return 10
        self.xml.write(element, pretty_print=self.prettyprint)

class GraphMLWriterLxml(GraphMLWriter):

    def __init__(self, path, graph=None, encoding='utf-8', prettyprint=True, infer_numeric_types=False, named_key_ids=False, edge_id_from_attribute=None):
        if False:
            for i in range(10):
                print('nop')
        self.construct_types()
        import lxml.etree as lxmletree
        self.myElement = lxmletree.Element
        self._encoding = encoding
        self._prettyprint = prettyprint
        self.named_key_ids = named_key_ids
        self.edge_id_from_attribute = edge_id_from_attribute
        self.infer_numeric_types = infer_numeric_types
        self._xml_base = lxmletree.xmlfile(path, encoding=encoding)
        self._xml = self._xml_base.__enter__()
        self._xml.write_declaration()
        self.xml = []
        self._keys = self.xml
        self._graphml = self._xml.element('graphml', {'xmlns': self.NS_GRAPHML, 'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation': self.SCHEMALOCATION})
        self._graphml.__enter__()
        self.keys = {}
        self.attribute_types = defaultdict(set)
        if graph is not None:
            self.add_graph_element(graph)

    def add_graph_element(self, G):
        if False:
            i = 10
            return i + 15
        '\n        Serialize graph G in GraphML to the stream.\n        '
        if G.is_directed():
            default_edge_type = 'directed'
        else:
            default_edge_type = 'undirected'
        graphid = G.graph.pop('id', None)
        if graphid is None:
            graph_element = self._xml.element('graph', edgedefault=default_edge_type)
        else:
            graph_element = self._xml.element('graph', edgedefault=default_edge_type, id=graphid)
        graphdata = {k: v for (k, v) in G.graph.items() if k not in ('node_default', 'edge_default')}
        node_default = G.graph.get('node_default', {})
        edge_default = G.graph.get('edge_default', {})
        for (k, v) in graphdata.items():
            self.attribute_types[str(k), 'graph'].add(type(v))
        for (k, v) in graphdata.items():
            element_type = self.get_xml_type(self.attr_type(k, 'graph', v))
            self.get_key(str(k), element_type, 'graph', None)
        for (node, d) in G.nodes(data=True):
            for (k, v) in d.items():
                self.attribute_types[str(k), 'node'].add(type(v))
        for (node, d) in G.nodes(data=True):
            for (k, v) in d.items():
                T = self.get_xml_type(self.attr_type(k, 'node', v))
                self.get_key(str(k), T, 'node', node_default.get(k))
        if G.is_multigraph():
            for (u, v, ekey, d) in G.edges(keys=True, data=True):
                for (k, v) in d.items():
                    self.attribute_types[str(k), 'edge'].add(type(v))
            for (u, v, ekey, d) in G.edges(keys=True, data=True):
                for (k, v) in d.items():
                    T = self.get_xml_type(self.attr_type(k, 'edge', v))
                    self.get_key(str(k), T, 'edge', edge_default.get(k))
        else:
            for (u, v, d) in G.edges(data=True):
                for (k, v) in d.items():
                    self.attribute_types[str(k), 'edge'].add(type(v))
            for (u, v, d) in G.edges(data=True):
                for (k, v) in d.items():
                    T = self.get_xml_type(self.attr_type(k, 'edge', v))
                    self.get_key(str(k), T, 'edge', edge_default.get(k))
        for key in self.xml:
            self._xml.write(key, pretty_print=self._prettyprint)
        incremental_writer = IncrementalElement(self._xml, self._prettyprint)
        with graph_element:
            self.add_attributes('graph', incremental_writer, graphdata, {})
            self.add_nodes(G, incremental_writer)
            self.add_edges(G, incremental_writer)

    def add_attributes(self, scope, xml_obj, data, default):
        if False:
            while True:
                i = 10
        'Appends attribute data.'
        for (k, v) in data.items():
            data_element = self.add_data(str(k), self.attr_type(str(k), scope, v), str(v), scope, default.get(k))
            xml_obj.append(data_element)

    def __str__(self):
        if False:
            return 10
        return object.__str__(self)

    def dump(self, stream=None):
        if False:
            return 10
        self._graphml.__exit__(None, None, None)
        self._xml_base.__exit__(None, None, None)
write_graphml = write_graphml_lxml

class GraphMLReader(GraphML):
    """Read a GraphML document.  Produces NetworkX graph objects."""

    def __init__(self, node_type=str, edge_key_type=int, force_multigraph=False):
        if False:
            i = 10
            return i + 15
        self.construct_types()
        self.node_type = node_type
        self.edge_key_type = edge_key_type
        self.multigraph = force_multigraph
        self.edge_ids = {}

    def __call__(self, path=None, string=None):
        if False:
            i = 10
            return i + 15
        from xml.etree.ElementTree import ElementTree, fromstring
        if path is not None:
            self.xml = ElementTree(file=path)
        elif string is not None:
            self.xml = fromstring(string)
        else:
            raise ValueError("Must specify either 'path' or 'string' as kwarg")
        (keys, defaults) = self.find_graphml_keys(self.xml)
        for g in self.xml.findall(f'{{{self.NS_GRAPHML}}}graph'):
            yield self.make_graph(g, keys, defaults)

    def make_graph(self, graph_xml, graphml_keys, defaults, G=None):
        if False:
            return 10
        edgedefault = graph_xml.get('edgedefault', None)
        if G is None:
            if edgedefault == 'directed':
                G = nx.MultiDiGraph()
            else:
                G = nx.MultiGraph()
        G.graph['node_default'] = {}
        G.graph['edge_default'] = {}
        for (key_id, value) in defaults.items():
            key_for = graphml_keys[key_id]['for']
            name = graphml_keys[key_id]['name']
            python_type = graphml_keys[key_id]['type']
            if key_for == 'node':
                G.graph['node_default'].update({name: python_type(value)})
            if key_for == 'edge':
                G.graph['edge_default'].update({name: python_type(value)})
        hyperedge = graph_xml.find(f'{{{self.NS_GRAPHML}}}hyperedge')
        if hyperedge is not None:
            raise nx.NetworkXError("GraphML reader doesn't support hyperedges")
        for node_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}node'):
            self.add_node(G, node_xml, graphml_keys, defaults)
        for edge_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}edge'):
            self.add_edge(G, edge_xml, graphml_keys)
        data = self.decode_data_elements(graphml_keys, graph_xml)
        G.graph.update(data)
        if self.multigraph:
            return G
        G = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)
        nx.set_edge_attributes(G, values=self.edge_ids, name='id')
        return G

    def add_node(self, G, node_xml, graphml_keys, defaults):
        if False:
            return 10
        'Add a node to the graph.'
        ports = node_xml.find(f'{{{self.NS_GRAPHML}}}port')
        if ports is not None:
            warnings.warn('GraphML port tag not supported.')
        node_id = self.node_type(node_xml.get('id'))
        data = self.decode_data_elements(graphml_keys, node_xml)
        G.add_node(node_id, **data)
        if node_xml.attrib.get('yfiles.foldertype') == 'group':
            graph_xml = node_xml.find(f'{{{self.NS_GRAPHML}}}graph')
            self.make_graph(graph_xml, graphml_keys, defaults, G)

    def add_edge(self, G, edge_element, graphml_keys):
        if False:
            for i in range(10):
                print('nop')
        'Add an edge to the graph.'
        ports = edge_element.find(f'{{{self.NS_GRAPHML}}}port')
        if ports is not None:
            warnings.warn('GraphML port tag not supported.')
        directed = edge_element.get('directed')
        if G.is_directed() and directed == 'false':
            msg = 'directed=false edge found in directed graph.'
            raise nx.NetworkXError(msg)
        if not G.is_directed() and directed == 'true':
            msg = 'directed=true edge found in undirected graph.'
            raise nx.NetworkXError(msg)
        source = self.node_type(edge_element.get('source'))
        target = self.node_type(edge_element.get('target'))
        data = self.decode_data_elements(graphml_keys, edge_element)
        edge_id = edge_element.get('id')
        if edge_id:
            self.edge_ids[source, target] = edge_id
            try:
                edge_id = self.edge_key_type(edge_id)
            except ValueError:
                pass
        else:
            edge_id = data.get('key')
        if G.has_edge(source, target):
            self.multigraph = True
        G.add_edges_from([(source, target, edge_id, data)])

    def decode_data_elements(self, graphml_keys, obj_xml):
        if False:
            return 10
        'Use the key information to decode the data XML if present.'
        data = {}
        for data_element in obj_xml.findall(f'{{{self.NS_GRAPHML}}}data'):
            key = data_element.get('key')
            try:
                data_name = graphml_keys[key]['name']
                data_type = graphml_keys[key]['type']
            except KeyError as err:
                raise nx.NetworkXError(f'Bad GraphML data: no key {key}') from err
            text = data_element.text
            if text is not None and len(list(data_element)) == 0:
                if data_type == bool:
                    data[data_name] = self.convert_bool[text.lower()]
                else:
                    data[data_name] = data_type(text)
            elif len(list(data_element)) > 0:
                node_label = None
                gn = data_element.find(f'{{{self.NS_Y}}}GenericNode')
                if gn:
                    data['shape_type'] = gn.get('configuration')
                for node_type in ['GenericNode', 'ShapeNode', 'SVGNode', 'ImageNode']:
                    pref = f'{{{self.NS_Y}}}{node_type}/{{{self.NS_Y}}}'
                    geometry = data_element.find(f'{pref}Geometry')
                    if geometry is not None:
                        data['x'] = geometry.get('x')
                        data['y'] = geometry.get('y')
                    if node_label is None:
                        node_label = data_element.find(f'{pref}NodeLabel')
                    shape = data_element.find(f'{pref}Shape')
                    if shape is not None:
                        data['shape_type'] = shape.get('type')
                if node_label is not None:
                    data['label'] = node_label.text
                for edge_type in ['PolyLineEdge', 'SplineEdge', 'QuadCurveEdge', 'BezierEdge', 'ArcEdge']:
                    pref = f'{{{self.NS_Y}}}{edge_type}/{{{self.NS_Y}}}'
                    edge_label = data_element.find(f'{pref}EdgeLabel')
                    if edge_label is not None:
                        break
                if edge_label is not None:
                    data['label'] = edge_label.text
        return data

    def find_graphml_keys(self, graph_element):
        if False:
            while True:
                i = 10
        'Extracts all the keys and key defaults from the xml.'
        graphml_keys = {}
        graphml_key_defaults = {}
        for k in graph_element.findall(f'{{{self.NS_GRAPHML}}}key'):
            attr_id = k.get('id')
            attr_type = k.get('attr.type')
            attr_name = k.get('attr.name')
            yfiles_type = k.get('yfiles.type')
            if yfiles_type is not None:
                attr_name = yfiles_type
                attr_type = 'yfiles'
            if attr_type is None:
                attr_type = 'string'
                warnings.warn(f'No key type for id {attr_id}. Using string')
            if attr_name is None:
                raise nx.NetworkXError(f'Unknown key for id {attr_id}.')
            graphml_keys[attr_id] = {'name': attr_name, 'type': self.python_type[attr_type], 'for': k.get('for')}
            default = k.find(f'{{{self.NS_GRAPHML}}}default')
            if default is not None:
                python_type = graphml_keys[attr_id]['type']
                if python_type == bool:
                    graphml_key_defaults[attr_id] = self.convert_bool[default.text.lower()]
                else:
                    graphml_key_defaults[attr_id] = python_type(default.text)
        return (graphml_keys, graphml_key_defaults)