import io
import time
import pytest
import networkx as nx

class TestGEXF:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.simple_directed_data = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n    <graph mode="static" defaultedgetype="directed">\n        <nodes>\n            <node id="0" label="Hello" />\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" />\n        </edges>\n    </graph>\n</gexf>\n'
        cls.simple_directed_graph = nx.DiGraph()
        cls.simple_directed_graph.add_node('0', label='Hello')
        cls.simple_directed_graph.add_node('1', label='World')
        cls.simple_directed_graph.add_edge('0', '1', id='0')
        cls.simple_directed_fh = io.BytesIO(cls.simple_directed_data.encode('UTF-8'))
        cls.attribute_data = '<?xml version="1.0" encoding="UTF-8"?><gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n  <meta lastmodifieddate="2009-03-20">\n    <creator>Gephi.org</creator>\n    <description>A Web network</description>\n  </meta>\n  <graph defaultedgetype="directed">\n    <attributes class="node">\n      <attribute id="0" title="url" type="string"/>\n      <attribute id="1" title="indegree" type="integer"/>\n      <attribute id="2" title="frog" type="boolean">\n        <default>true</default>\n      </attribute>\n    </attributes>\n    <nodes>\n      <node id="0" label="Gephi">\n        <attvalues>\n          <attvalue for="0" value="https://gephi.org"/>\n          <attvalue for="1" value="1"/>\n          <attvalue for="2" value="false"/>\n        </attvalues>\n      </node>\n      <node id="1" label="Webatlas">\n        <attvalues>\n          <attvalue for="0" value="http://webatlas.fr"/>\n          <attvalue for="1" value="2"/>\n          <attvalue for="2" value="false"/>\n        </attvalues>\n      </node>\n      <node id="2" label="RTGI">\n        <attvalues>\n          <attvalue for="0" value="http://rtgi.fr"/>\n          <attvalue for="1" value="1"/>\n          <attvalue for="2" value="true"/>\n        </attvalues>\n      </node>\n      <node id="3" label="BarabasiLab">\n        <attvalues>\n          <attvalue for="0" value="http://barabasilab.com"/>\n          <attvalue for="1" value="1"/>\n          <attvalue for="2" value="true"/>\n        </attvalues>\n      </node>\n    </nodes>\n    <edges>\n      <edge id="0" source="0" target="1" label="foo"/>\n      <edge id="1" source="0" target="2"/>\n      <edge id="2" source="1" target="0"/>\n      <edge id="3" source="2" target="1"/>\n      <edge id="4" source="0" target="3"/>\n    </edges>\n  </graph>\n</gexf>\n'
        cls.attribute_graph = nx.DiGraph()
        cls.attribute_graph.graph['node_default'] = {'frog': True}
        cls.attribute_graph.add_node('0', label='Gephi', url='https://gephi.org', indegree=1, frog=False)
        cls.attribute_graph.add_node('1', label='Webatlas', url='http://webatlas.fr', indegree=2, frog=False)
        cls.attribute_graph.add_node('2', label='RTGI', url='http://rtgi.fr', indegree=1, frog=True)
        cls.attribute_graph.add_node('3', label='BarabasiLab', url='http://barabasilab.com', indegree=1, frog=True)
        cls.attribute_graph.add_edge('0', '1', id='0', label='foo')
        cls.attribute_graph.add_edge('0', '2', id='1')
        cls.attribute_graph.add_edge('1', '0', id='2')
        cls.attribute_graph.add_edge('2', '1', id='3')
        cls.attribute_graph.add_edge('0', '3', id='4')
        cls.attribute_fh = io.BytesIO(cls.attribute_data.encode('UTF-8'))
        cls.simple_undirected_data = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n    <graph mode="static" defaultedgetype="undirected">\n        <nodes>\n            <node id="0" label="Hello" />\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" />\n        </edges>\n    </graph>\n</gexf>\n'
        cls.simple_undirected_graph = nx.Graph()
        cls.simple_undirected_graph.add_node('0', label='Hello')
        cls.simple_undirected_graph.add_node('1', label='World')
        cls.simple_undirected_graph.add_edge('0', '1', id='0')
        cls.simple_undirected_fh = io.BytesIO(cls.simple_undirected_data.encode('UTF-8'))

    def test_read_simple_directed_graphml(self):
        if False:
            while True:
                i = 10
        G = self.simple_directed_graph
        H = nx.read_gexf(self.simple_directed_fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_write_read_simple_directed_graphml(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.simple_directed_graph
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_read_simple_undirected_graphml(self):
        if False:
            while True:
                i = 10
        G = self.simple_undirected_graph
        H = nx.read_gexf(self.simple_undirected_fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
        self.simple_undirected_fh.seek(0)

    def test_read_attribute_graphml(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.attribute_graph
        H = nx.read_gexf(self.attribute_fh)
        assert sorted(G.nodes(True)) == sorted(H.nodes(data=True))
        ge = sorted(G.edges(data=True))
        he = sorted(H.edges(data=True))
        for (a, b) in zip(ge, he):
            assert a == b
        self.attribute_fh.seek(0)

    def test_directed_edge_in_undirected(self):
        if False:
            while True:
                i = 10
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version=\'1.2\'>\n    <graph mode="static" defaultedgetype="undirected" name="">\n        <nodes>\n            <node id="0" label="Hello" />\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" type="directed"/>\n        </edges>\n    </graph>\n</gexf>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_gexf, fh)

    def test_undirected_edge_in_directed(self):
        if False:
            while True:
                i = 10
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version=\'1.2\'>\n    <graph mode="static" defaultedgetype="directed" name="">\n        <nodes>\n            <node id="0" label="Hello" />\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" type="undirected"/>\n        </edges>\n    </graph>\n</gexf>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_gexf, fh)

    def test_key_raises(self):
        if False:
            i = 10
            return i + 15
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version=\'1.2\'>\n    <graph mode="static" defaultedgetype="directed" name="">\n        <nodes>\n            <node id="0" label="Hello">\n              <attvalues>\n                <attvalue for=\'0\' value=\'1\'/>\n              </attvalues>\n            </node>\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1" type="undirected"/>\n        </edges>\n    </graph>\n</gexf>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_gexf, fh)

    def test_relabel(self):
        if False:
            for i in range(10):
                print('nop')
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<gexf xmlns="http://www.gexf.net/1.2draft" version=\'1.2\'>\n    <graph mode="static" defaultedgetype="directed" name="">\n        <nodes>\n            <node id="0" label="Hello" />\n            <node id="1" label="Word" />\n        </nodes>\n        <edges>\n            <edge id="0" source="0" target="1"/>\n        </edges>\n    </graph>\n</gexf>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_gexf(fh, relabel=True)
        assert sorted(G.nodes()) == ['Hello', 'Word']

    def test_default_attribute(self):
        if False:
            return 10
        G = nx.Graph()
        G.add_node(1, label='1', color='green')
        nx.add_path(G, [0, 1, 2, 3])
        G.add_edge(1, 2, foo=3)
        G.graph['node_default'] = {'color': 'yellow'}
        G.graph['edge_default'] = {'foo': 7}
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
        del H.graph['mode']
        assert G.graph == H.graph

    def test_serialize_ints_to_strings(self):
        if False:
            while True:
                i = 10
        G = nx.Graph()
        G.add_node(1, id=7, label=77)
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert list(H) == [7]
        assert H.nodes[7]['label'] == '77'

    def test_write_with_node_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for i in range(4):
            G.nodes[i]['id'] = i
            G.nodes[i]['label'] = i
            G.nodes[i]['pid'] = i
            G.nodes[i]['start'] = i
            G.nodes[i]['end'] = i + 1
        expected = f'''<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">\n    <creator>NetworkX {nx.__version__}</creator>\n  </meta>\n  <graph defaultedgetype="undirected" mode="dynamic" name="" timeformat="long">\n    <nodes>\n      <node id="0" label="0" pid="0" start="0" end="1" />\n      <node id="1" label="1" pid="1" start="1" end="2" />\n      <node id="2" label="2" pid="2" start="2" end="3" />\n      <node id="3" label="3" pid="3" start="3" end="4" />\n    </nodes>\n    <edges>\n      <edge source="0" target="1" id="0" />\n      <edge source="1" target="2" id="1" />\n      <edge source="2" target="3" id="2" />\n    </edges>\n  </graph>\n</gexf>'''
        obtained = '\n'.join(nx.generate_gexf(G))
        assert expected == obtained

    def test_edge_id_construct(self):
        if False:
            i = 10
            return i + 15
        G = nx.Graph()
        G.add_edges_from([(0, 1, {'id': 0}), (1, 2, {'id': 2}), (2, 3)])
        expected = f'''<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">\n    <creator>NetworkX {nx.__version__}</creator>\n  </meta>\n  <graph defaultedgetype="undirected" mode="static" name="">\n    <nodes>\n      <node id="0" label="0" />\n      <node id="1" label="1" />\n      <node id="2" label="2" />\n      <node id="3" label="3" />\n    </nodes>\n    <edges>\n      <edge source="0" target="1" id="0" />\n      <edge source="1" target="2" id="2" />\n      <edge source="2" target="3" id="1" />\n    </edges>\n  </graph>\n</gexf>'''
        obtained = '\n'.join(nx.generate_gexf(G))
        assert expected == obtained

    def test_numpy_type(self):
        if False:
            while True:
                i = 10
        np = pytest.importorskip('numpy')
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {n: n for n in np.arange(4)}, 'number')
        G[0][1]['edge-number'] = np.float64(1.1)
        expected = f'''<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">\n  <meta lastmodifieddate="{time.strftime('%Y-%m-%d')}">\n    <creator>NetworkX {nx.__version__}</creator>\n  </meta>\n  <graph defaultedgetype="undirected" mode="static" name="">\n    <attributes mode="static" class="edge">\n      <attribute id="1" title="edge-number" type="float" />\n    </attributes>\n    <attributes mode="static" class="node">\n      <attribute id="0" title="number" type="int" />\n    </attributes>\n    <nodes>\n      <node id="0" label="0">\n        <attvalues>\n          <attvalue for="0" value="0" />\n        </attvalues>\n      </node>\n      <node id="1" label="1">\n        <attvalues>\n          <attvalue for="0" value="1" />\n        </attvalues>\n      </node>\n      <node id="2" label="2">\n        <attvalues>\n          <attvalue for="0" value="2" />\n        </attvalues>\n      </node>\n      <node id="3" label="3">\n        <attvalues>\n          <attvalue for="0" value="3" />\n        </attvalues>\n      </node>\n    </nodes>\n    <edges>\n      <edge source="0" target="1" id="0">\n        <attvalues>\n          <attvalue for="1" value="1.1" />\n        </attvalues>\n      </edge>\n      <edge source="1" target="2" id="1" />\n      <edge source="2" target="3" id="2" />\n    </edges>\n  </graph>\n</gexf>'''
        obtained = '\n'.join(nx.generate_gexf(G))
        assert expected == obtained

    def test_bool(self):
        if False:
            return 10
        G = nx.Graph()
        G.add_node(1, testattr=True)
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert H.nodes[1]['testattr']

    def test_specials(self):
        if False:
            return 10
        from math import isnan
        (inf, nan) = (float('inf'), float('nan'))
        G = nx.Graph()
        G.add_node(1, testattr=inf, strdata='inf', key='a')
        G.add_node(2, testattr=nan, strdata='nan', key='b')
        G.add_node(3, testattr=-inf, strdata='-inf', key='c')
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        filetext = fh.read()
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert b'INF' in filetext
        assert b'NaN' in filetext
        assert b'-INF' in filetext
        assert H.nodes[1]['testattr'] == inf
        assert isnan(H.nodes[2]['testattr'])
        assert H.nodes[3]['testattr'] == -inf
        assert H.nodes[1]['strdata'] == 'inf'
        assert H.nodes[2]['strdata'] == 'nan'
        assert H.nodes[3]['strdata'] == '-inf'
        assert H.nodes[1]['networkx_key'] == 'a'
        assert H.nodes[2]['networkx_key'] == 'b'
        assert H.nodes[3]['networkx_key'] == 'c'

    def test_simple_list(self):
        if False:
            i = 10
            return i + 15
        G = nx.Graph()
        list_value = [(1, 2, 3), (9, 1, 2)]
        G.add_node(1, key=list_value)
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert H.nodes[1]['networkx_key'] == list_value

    def test_dynamic_mode(self):
        if False:
            while True:
                i = 10
        G = nx.Graph()
        G.add_node(1, label='1', color='green')
        G.graph['mode'] = 'dynamic'
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))

    def test_multigraph_with_missing_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.MultiGraph()
        G.add_node(0, label='1', color='green')
        G.add_node(1, label='2', color='green')
        G.add_edge(0, 1, id='0', weight=3, type='undirected', start=0, end=1)
        G.add_edge(0, 1, id='1', label='foo', start=0, end=1)
        G.add_edge(0, 1)
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))

    def test_missing_viz_attributes(self):
        if False:
            print('Hello World!')
        G = nx.Graph()
        G.add_node(0, label='1', color='green')
        G.nodes[0]['viz'] = {'size': 54}
        G.nodes[0]['viz']['position'] = {'x': 0, 'y': 1, 'z': 0}
        G.nodes[0]['viz']['color'] = {'r': 0, 'g': 0, 'b': 256}
        G.nodes[0]['viz']['shape'] = 'http://random.url'
        G.nodes[0]['viz']['thickness'] = 2
        fh = io.BytesIO()
        nx.write_gexf(G, fh, version='1.1draft')
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
        fh = io.BytesIO()
        nx.write_gexf(G, fh, version='1.2draft')
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert H.nodes[0]['viz']['color']['a'] == 1.0
        G = nx.Graph()
        G.add_node(0, label='1', color='green')
        G.nodes[0]['viz'] = {'size': 54}
        G.nodes[0]['viz']['position'] = {'x': 0, 'y': 1, 'z': 0}
        G.nodes[0]['viz']['color'] = {'r': 0, 'g': 0, 'b': 256, 'a': 0.5}
        G.nodes[0]['viz']['shape'] = 'ftp://random.url'
        G.nodes[0]['viz']['thickness'] = 2
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))

    def test_slice_and_spell(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph()
        G.add_node(0, label='1', color='green')
        G.nodes[0]['spells'] = [(1, 2)]
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
        G = nx.Graph()
        G.add_node(0, label='1', color='green')
        G.nodes[0]['slices'] = [(1, 2)]
        fh = io.BytesIO()
        nx.write_gexf(G, fh, version='1.1draft')
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))

    def test_add_parent(self):
        if False:
            while True:
                i = 10
        G = nx.Graph()
        G.add_node(0, label='1', color='green', parents=[1, 2])
        fh = io.BytesIO()
        nx.write_gexf(G, fh)
        fh.seek(0)
        H = nx.read_gexf(fh, node_type=int)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))