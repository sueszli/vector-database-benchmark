import codecs
import io
import math
import os
import tempfile
from ast import literal_eval
from contextlib import contextmanager
from textwrap import dedent
import pytest
import networkx as nx
from networkx.readwrite.gml import literal_destringizer, literal_stringizer

class TestGraph:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.simple_data = 'Creator "me"\nVersion "xx"\ngraph [\n comment "This is a sample graph"\n directed 1\n IsPlanar 1\n pos  [ x 0 y 1 ]\n node [\n   id 1\n   label "Node 1"\n   pos [ x 1 y 1 ]\n ]\n node [\n    id 2\n    pos [ x 1 y 2 ]\n    label "Node 2"\n    ]\n  node [\n    id 3\n    label "Node 3"\n    pos [ x 1 y 3 ]\n  ]\n  edge [\n    source 1\n    target 2\n    label "Edge from node 1 to node 2"\n    color [line "blue" thickness 3]\n\n  ]\n  edge [\n    source 2\n    target 3\n    label "Edge from node 2 to node 3"\n  ]\n  edge [\n    source 3\n    target 1\n    label "Edge from node 3 to node 1"\n  ]\n]\n'

    def test_parse_gml_cytoscape_bug(self):
        if False:
            i = 10
            return i + 15
        cytoscape_example = '\nCreator "Cytoscape"\nVersion 1.0\ngraph   [\n    node    [\n        root_index  -3\n        id  -3\n        graphics    [\n            x   -96.0\n            y   -67.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node2"\n    ]\n    node    [\n        root_index  -2\n        id  -2\n        graphics    [\n            x   63.0\n            y   37.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node1"\n    ]\n    node    [\n        root_index  -1\n        id  -1\n        graphics    [\n            x   -31.0\n            y   -17.0\n            w   40.0\n            h   40.0\n            fill    "#ff9999"\n            type    "ellipse"\n            outline "#666666"\n            outline_width   1.5\n        ]\n        label   "node0"\n    ]\n    edge    [\n        root_index  -2\n        target  -2\n        source  -1\n        graphics    [\n            width   1.5\n            fill    "#0000ff"\n            type    "line"\n            Line    [\n            ]\n            source_arrow    0\n            target_arrow    3\n        ]\n        label   "DirectedEdge"\n    ]\n    edge    [\n        root_index  -1\n        target  -1\n        source  -3\n        graphics    [\n            width   1.5\n            fill    "#0000ff"\n            type    "line"\n            Line    [\n            ]\n            source_arrow    0\n            target_arrow    3\n        ]\n        label   "DirectedEdge"\n    ]\n]\n'
        nx.parse_gml(cytoscape_example)

    def test_parse_gml(self):
        if False:
            return 10
        G = nx.parse_gml(self.simple_data, label='label')
        assert sorted(G.nodes()) == ['Node 1', 'Node 2', 'Node 3']
        assert sorted(G.edges()) == [('Node 1', 'Node 2'), ('Node 2', 'Node 3'), ('Node 3', 'Node 1')]
        assert sorted(G.edges(data=True)) == [('Node 1', 'Node 2', {'color': {'line': 'blue', 'thickness': 3}, 'label': 'Edge from node 1 to node 2'}), ('Node 2', 'Node 3', {'label': 'Edge from node 2 to node 3'}), ('Node 3', 'Node 1', {'label': 'Edge from node 3 to node 1'})]

    def test_read_gml(self):
        if False:
            for i in range(10):
                print('nop')
        (fd, fname) = tempfile.mkstemp()
        fh = open(fname, 'w')
        fh.write(self.simple_data)
        fh.close()
        Gin = nx.read_gml(fname, label='label')
        G = nx.parse_gml(self.simple_data, label='label')
        assert sorted(G.nodes(data=True)) == sorted(Gin.nodes(data=True))
        assert sorted(G.edges(data=True)) == sorted(Gin.edges(data=True))
        os.close(fd)
        os.unlink(fname)

    def test_labels_are_strings(self):
        if False:
            return 10
        answer = 'graph [\n  node [\n    id 0\n    label "1203"\n  ]\n]'
        G = nx.Graph()
        G.add_node(1203)
        data = '\n'.join(nx.generate_gml(G, stringizer=literal_stringizer))
        assert data == answer

    def test_relabel_duplicate(self):
        if False:
            while True:
                i = 10
        data = '\ngraph\n[\n        label   ""\n        directed        1\n        node\n        [\n                id      0\n                label   "same"\n        ]\n        node\n        [\n                id      1\n                label   "same"\n        ]\n]\n'
        fh = io.BytesIO(data.encode('UTF-8'))
        fh.seek(0)
        pytest.raises(nx.NetworkXError, nx.read_gml, fh, label='label')

    @pytest.mark.parametrize('stringizer', (None, literal_stringizer))
    def test_tuplelabels(self, stringizer):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph()
        G.add_edge((0, 1), (1, 0))
        data = '\n'.join(nx.generate_gml(G, stringizer=stringizer))
        answer = 'graph [\n  node [\n    id 0\n    label "(0,1)"\n  ]\n  node [\n    id 1\n    label "(1,0)"\n  ]\n  edge [\n    source 0\n    target 1\n  ]\n]'
        assert data == answer

    def test_quotes(self):
        if False:
            i = 10
            return i + 15
        G = nx.path_graph(1)
        G.name = 'path_graph(1)'
        attr = 'This is "quoted" and this is a copyright: ' + chr(169)
        G.nodes[0]['demo'] = attr
        fobj = tempfile.NamedTemporaryFile()
        nx.write_gml(G, fobj)
        fobj.seek(0)
        data = fobj.read().strip().decode('ascii')
        answer = 'graph [\n  name "path_graph(1)"\n  node [\n    id 0\n    label "0"\n    demo "This is &#34;quoted&#34; and this is a copyright: &#169;"\n  ]\n]'
        assert data == answer

    def test_unicode_node(self):
        if False:
            while True:
                i = 10
        node = 'node' + chr(169)
        G = nx.Graph()
        G.add_node(node)
        fobj = tempfile.NamedTemporaryFile()
        nx.write_gml(G, fobj)
        fobj.seek(0)
        data = fobj.read().strip().decode('ascii')
        answer = 'graph [\n  node [\n    id 0\n    label "node&#169;"\n  ]\n]'
        assert data == answer

    def test_float_label(self):
        if False:
            return 10
        node = 1.0
        G = nx.Graph()
        G.add_node(node)
        fobj = tempfile.NamedTemporaryFile()
        nx.write_gml(G, fobj)
        fobj.seek(0)
        data = fobj.read().strip().decode('ascii')
        answer = 'graph [\n  node [\n    id 0\n    label "1.0"\n  ]\n]'
        assert data == answer

    def test_special_float_label(self):
        if False:
            return 10
        special_floats = [float('nan'), float('+inf'), float('-inf')]
        try:
            import numpy as np
            special_floats += [np.nan, np.inf, np.inf * -1]
        except ImportError:
            special_floats += special_floats
        G = nx.cycle_graph(len(special_floats))
        attrs = dict(enumerate(special_floats))
        nx.set_node_attributes(G, attrs, 'nodefloat')
        edges = list(G.edges)
        attrs = {edges[i]: value for (i, value) in enumerate(special_floats)}
        nx.set_edge_attributes(G, attrs, 'edgefloat')
        fobj = tempfile.NamedTemporaryFile()
        nx.write_gml(G, fobj)
        fobj.seek(0)
        data = fobj.read().strip().decode('ascii')
        answer = 'graph [\n  node [\n    id 0\n    label "0"\n    nodefloat NAN\n  ]\n  node [\n    id 1\n    label "1"\n    nodefloat +INF\n  ]\n  node [\n    id 2\n    label "2"\n    nodefloat -INF\n  ]\n  node [\n    id 3\n    label "3"\n    nodefloat NAN\n  ]\n  node [\n    id 4\n    label "4"\n    nodefloat +INF\n  ]\n  node [\n    id 5\n    label "5"\n    nodefloat -INF\n  ]\n  edge [\n    source 0\n    target 1\n    edgefloat NAN\n  ]\n  edge [\n    source 0\n    target 5\n    edgefloat +INF\n  ]\n  edge [\n    source 1\n    target 2\n    edgefloat -INF\n  ]\n  edge [\n    source 2\n    target 3\n    edgefloat NAN\n  ]\n  edge [\n    source 3\n    target 4\n    edgefloat +INF\n  ]\n  edge [\n    source 4\n    target 5\n    edgefloat -INF\n  ]\n]'
        assert data == answer
        fobj.seek(0)
        graph = nx.read_gml(fobj)
        for (indx, value) in enumerate(special_floats):
            node_value = graph.nodes[str(indx)]['nodefloat']
            if math.isnan(value):
                assert math.isnan(node_value)
            else:
                assert node_value == value
            edge = edges[indx]
            string_edge = (str(edge[0]), str(edge[1]))
            edge_value = graph.edges[string_edge]['edgefloat']
            if math.isnan(value):
                assert math.isnan(edge_value)
            else:
                assert edge_value == value

    def test_name(self):
        if False:
            i = 10
            return i + 15
        G = nx.parse_gml('graph [ name "x" node [ id 0 label "x" ] ]')
        assert 'x' == G.graph['name']
        G = nx.parse_gml('graph [ node [ id 0 label "x" ] ]')
        assert '' == G.name
        assert 'name' not in G.graph

    def test_graph_types(self):
        if False:
            print('Hello World!')
        for directed in [None, False, True]:
            for multigraph in [None, False, True]:
                gml = 'graph ['
                if directed is not None:
                    gml += ' directed ' + str(int(directed))
                if multigraph is not None:
                    gml += ' multigraph ' + str(int(multigraph))
                gml += ' node [ id 0 label "0" ]'
                gml += ' edge [ source 0 target 0 ]'
                gml += ' ]'
                G = nx.parse_gml(gml)
                assert bool(directed) == G.is_directed()
                assert bool(multigraph) == G.is_multigraph()
                gml = 'graph [\n'
                if directed is True:
                    gml += '  directed 1\n'
                if multigraph is True:
                    gml += '  multigraph 1\n'
                gml += '  node [\n    id 0\n    label "0"\n  ]\n  edge [\n    source 0\n    target 0\n'
                if multigraph:
                    gml += '    key 0\n'
                gml += '  ]\n]'
                assert gml == '\n'.join(nx.generate_gml(G))

    def test_data_types(self):
        if False:
            print('Hello World!')
        data = [True, False, 10 ** 20, -2e+33, "'", '"&&amp;&&#34;"', [{(b'\xfd',): '\x7f', chr(17476): (1, 2)}, (2, '3')]]
        data.append(chr(83012))
        data.append(literal_eval('{2.3j, 1 - 2.3j, ()}'))
        G = nx.Graph()
        G.name = data
        G.graph['data'] = data
        G.add_node(0, int=-1, data={'data': data})
        G.add_edge(0, 0, float=-2.5, data=data)
        gml = '\n'.join(nx.generate_gml(G, stringizer=literal_stringizer))
        G = nx.parse_gml(gml, destringizer=literal_destringizer)
        assert data == G.name
        assert {'name': data, 'data': data} == G.graph
        assert list(G.nodes(data=True)) == [(0, {'int': -1, 'data': {'data': data}})]
        assert list(G.edges(data=True)) == [(0, 0, {'float': -2.5, 'data': data})]
        G = nx.Graph()
        G.graph['data'] = 'frozenset([1, 2, 3])'
        G = nx.parse_gml(nx.generate_gml(G), destringizer=literal_eval)
        assert G.graph['data'] == 'frozenset([1, 2, 3])'

    def test_escape_unescape(self):
        if False:
            for i in range(10):
                print('nop')
        gml = 'graph [\n  name "&amp;&#34;&#xf;&#x4444;&#1234567890;&#x1234567890abcdef;&unknown;"\n]'
        G = nx.parse_gml(gml)
        assert '&"\x0f' + chr(17476) + '&#1234567890;&#x1234567890abcdef;&unknown;' == G.name
        gml = '\n'.join(nx.generate_gml(G))
        alnu = '#1234567890;&#38;#x1234567890abcdef'
        answer = 'graph [\n  name "&#38;&#34;&#15;&#17476;&#38;' + alnu + ';&#38;unknown;"\n]'
        assert answer == gml

    def test_exceptions(self):
        if False:
            return 10
        pytest.raises(ValueError, literal_destringizer, '(')
        pytest.raises(ValueError, literal_destringizer, 'frozenset([1, 2, 3])')
        pytest.raises(ValueError, literal_destringizer, literal_destringizer)
        pytest.raises(ValueError, literal_stringizer, frozenset([1, 2, 3]))
        pytest.raises(ValueError, literal_stringizer, literal_stringizer)
        with tempfile.TemporaryFile() as f:
            f.write(codecs.BOM_UTF8 + b'graph[]')
            f.seek(0)
            pytest.raises(nx.NetworkXError, nx.read_gml, f)

        def assert_parse_error(gml):
            if False:
                return 10
            pytest.raises(nx.NetworkXError, nx.parse_gml, gml)
        assert_parse_error(['graph [\n\n', ']'])
        assert_parse_error('')
        assert_parse_error('Creator ""')
        assert_parse_error('0')
        assert_parse_error('graph ]')
        assert_parse_error('graph [ 1 ]')
        assert_parse_error('graph [ 1.E+2 ]')
        assert_parse_error('graph [ "A" ]')
        assert_parse_error('graph [ ] graph ]')
        assert_parse_error('graph [ ] graph [ ]')
        assert_parse_error('graph [ data [1, 2, 3] ]')
        assert_parse_error('graph [ node [ ] ]')
        assert_parse_error('graph [ node [ id 0 ] ]')
        nx.parse_gml('graph [ node [ id "a" ] ]', label='id')
        assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 0 label 1 ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 0 ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] edge [ ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 0 ] ]')
        nx.parse_gml('graph [edge [ source 0 target 0 ] node [ id 0 label 0 ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 1 target 0 ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] edge [ source 0 target 1 ] ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 1 target 0 ] ]')
        nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 1 target 0 ] directed 1 ]')
        nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 ] edge [ source 0 target 1 ]multigraph 1 ]')
        nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 0 target 1 ]multigraph 1 ]')
        assert_parse_error('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 0 target 1 key 0 ]multigraph 1 ]')
        nx.parse_gml('graph [ node [ id 0 label 0 ] node [ id 1 label 1 ] edge [ source 0 target 1 key 0 ] edge [ source 1 target 0 key 0 ]directed 1 multigraph 1 ]')
        nx.parse_gml('graph [edge [ source a target a ] node [ id a label b ] ]')
        nx.parse_gml('graph [ node [ id n42 label 0 ] node [ id x43 label 1 ]edge [ source n42 target x43 key 0 ]edge [ source x43 target n42 key 0 ]directed 1 multigraph 1 ]')
        assert_parse_error("graph [edge [ source u'uĐ0' target u'uĐ0' ] " + "node [ id u'uĐ0' label b ] ]")

        def assert_generate_error(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            pytest.raises(nx.NetworkXError, lambda : list(nx.generate_gml(*args, **kwargs)))
        G = nx.Graph()
        G.graph[3] = 3
        assert_generate_error(G)
        G = nx.Graph()
        G.graph['3'] = 3
        assert_generate_error(G)
        G = nx.Graph()
        G.graph['data'] = frozenset([1, 2, 3])
        assert_generate_error(G, stringizer=literal_stringizer)

    def test_label_kwarg(self):
        if False:
            i = 10
            return i + 15
        G = nx.parse_gml(self.simple_data, label='id')
        assert sorted(G.nodes) == [1, 2, 3]
        labels = [G.nodes[n]['label'] for n in sorted(G.nodes)]
        assert labels == ['Node 1', 'Node 2', 'Node 3']
        G = nx.parse_gml(self.simple_data, label=None)
        assert sorted(G.nodes) == [1, 2, 3]
        labels = [G.nodes[n]['label'] for n in sorted(G.nodes)]
        assert labels == ['Node 1', 'Node 2', 'Node 3']

    def test_outofrange_integers(self):
        if False:
            print('Hello World!')
        G = nx.Graph()
        numbers = {'toosmall': -2 ** 31 - 1, 'small': -2 ** 31, 'med1': -4, 'med2': 0, 'med3': 17, 'big': 2 ** 31 - 1, 'toobig': 2 ** 31}
        G.add_node('Node', **numbers)
        (fd, fname) = tempfile.mkstemp()
        try:
            nx.write_gml(G, fname)
            G2 = nx.read_gml(fname)
            for (attr, value) in G2.nodes['Node'].items():
                if attr == 'toosmall' or attr == 'toobig':
                    assert type(value) == str
                else:
                    assert type(value) == int
        finally:
            os.close(fd)
            os.unlink(fname)

    def test_multiline(self):
        if False:
            i = 10
            return i + 15
        multiline_example = '\ngraph\n[\n    node\n    [\n\t    id 0\n\t    label "multiline node"\n\t    label2 "multiline1\n    multiline2\n    multiline3"\n\t    alt_name "id 0"\n    ]\n]\n'
        G = nx.parse_gml(multiline_example)
        assert G.nodes['multiline node'] == {'label2': 'multiline1 multiline2 multiline3', 'alt_name': 'id 0'}

@contextmanager
def byte_file():
    if False:
        i = 10
        return i + 15
    _file_handle = io.BytesIO()
    yield _file_handle
    _file_handle.seek(0)

class TestPropertyLists:

    def test_writing_graph_with_multi_element_property_list(self):
        if False:
            return 10
        g = nx.Graph()
        g.add_node('n1', properties=['element', 0, 1, 2.5, True, False])
        with byte_file() as f:
            nx.write_gml(g, f)
        result = f.read().decode()
        assert result == dedent('            graph [\n              node [\n                id 0\n                label "n1"\n                properties "element"\n                properties 0\n                properties 1\n                properties 2.5\n                properties 1\n                properties 0\n              ]\n            ]\n        ')

    def test_writing_graph_with_one_element_property_list(self):
        if False:
            for i in range(10):
                print('nop')
        g = nx.Graph()
        g.add_node('n1', properties=['element'])
        with byte_file() as f:
            nx.write_gml(g, f)
        result = f.read().decode()
        assert result == dedent('            graph [\n              node [\n                id 0\n                label "n1"\n                properties "_networkx_list_start"\n                properties "element"\n              ]\n            ]\n        ')

    def test_reading_graph_with_list_property(self):
        if False:
            print('Hello World!')
        with byte_file() as f:
            f.write(dedent('\n              graph [\n                node [\n                  id 0\n                  label "n1"\n                  properties "element"\n                  properties 0\n                  properties 1\n                  properties 2.5\n                ]\n              ]\n            ').encode('ascii'))
            f.seek(0)
            graph = nx.read_gml(f)
        assert graph.nodes(data=True)['n1'] == {'properties': ['element', 0, 1, 2.5]}

    def test_reading_graph_with_single_element_list_property(self):
        if False:
            for i in range(10):
                print('nop')
        with byte_file() as f:
            f.write(dedent('\n              graph [\n                node [\n                  id 0\n                  label "n1"\n                  properties "_networkx_list_start"\n                  properties "element"\n                ]\n              ]\n            ').encode('ascii'))
            f.seek(0)
            graph = nx.read_gml(f)
        assert graph.nodes(data=True)['n1'] == {'properties': ['element']}

@pytest.mark.parametrize('coll', ([], ()))
def test_stringize_empty_list_tuple(coll):
    if False:
        while True:
            i = 10
    G = nx.path_graph(2)
    G.nodes[0]['test'] = coll
    f = io.BytesIO()
    nx.write_gml(G, f)
    f.seek(0)
    H = nx.read_gml(f)
    assert H.nodes['0']['test'] == coll
    H = nx.relabel_nodes(H, {'0': 0, '1': 1})
    assert nx.utils.graphs_equal(G, H)
    f.seek(0)
    H = nx.read_gml(f, destringizer=int)
    assert nx.utils.graphs_equal(G, H)