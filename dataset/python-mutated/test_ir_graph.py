import unittest
from paddle import base

class TestIRGraph(unittest.TestCase):
    """
    TODO(fc500110): `resolve_hazard` api will be tested when it can be used.
    """

    def test_nodes(self):
        if False:
            print('Hello World!')
        graph = build_graph()
        self.assertTrue({node.name() for node in graph.nodes()} == {'x1', 'x2', 'out', 'sum'})

    def test_has_set_get(self):
        if False:
            for i in range(10):
                print('nop')
        graph = build_graph()
        for attr_name in ['int', 'float', 'string']:
            self.assertFalse(graph.has(attr_name))
        graph.set('int', 1)
        graph.set('float', 0.5)
        graph.set('string', 'string')
        for attr_name in ['int', 'float', 'string']:
            self.assertTrue(graph.has(attr_name))
        self.assertTrue(graph.get_int('int') == 1)
        self.assertTrue(graph.get_float('float') == 0.5)
        self.assertTrue(graph.get_string('string') == 'string')

    def test_erase(self):
        if False:
            print('Hello World!')
        graph = build_graph()
        graph.set('test', 0)
        self.assertTrue(graph.has('test'))
        graph.erase('test')
        self.assertFalse(graph.has('test'))

    def test_create_var_node(self):
        if False:
            for i in range(10):
                print('nop')
        prog = base.core.ProgramDesc()
        block = prog.block(0)
        shape = [10, 20]
        x1 = block.var(b'x1')
        x1.set_type(base.core.VarDesc.VarType.LOD_TENSOR)
        x1.set_shape(shape)
        graph = base.core.Graph(prog)
        node = graph.create_var_node(x1)
        self.assertTrue(node.node_type() == base.core.Node.Type.Variable)

    def test_create_op_node(self):
        if False:
            return 10
        prog = base.core.ProgramDesc()
        block = prog.block(0)
        sum_op_desc = block.append_op()
        graph = base.core.Graph(prog)
        node = graph.create_op_node(sum_op_desc)
        self.assertTrue(node.node_type() == base.core.Node.Type.Operation)

    def test_create_control_dep_var(self):
        if False:
            return 10
        graph = build_graph()
        name = f'__control_var@{len(graph.nodes())}'
        node = graph.create_control_dep_var()
        self.assertTrue(node.name() == name)

    def test_create_empty_node(self):
        if False:
            while True:
                i = 10
        prog = base.core.ProgramDesc()
        graph = base.core.Graph(prog)
        n1 = graph.create_empty_node('x', base.core.Node.Type.Operation)
        self.assertTrue(n1.name() == 'x')
        n2 = graph.create_empty_node('y', base.core.Node.Type.Variable)
        self.assertTrue(n2.name() == 'y')

    def test_release_nodes(self):
        if False:
            i = 10
            return i + 15
        graph = build_graph()
        nodes = graph.release_nodes()
        self.assertTrue(len(graph.nodes()) == 0)
        self.assertTrue({node.name() for node in nodes} == {'x1', 'x2', 'out', 'sum'})

    def test_remove_node(self):
        if False:
            i = 10
            return i + 15
        graph = build_graph()
        nodes = graph.nodes()
        for node in nodes:
            if node.name() == 'sum':
                break
        self.assertTrue({node.name() for node in nodes} == {'x1', 'x2', 'out', 'sum'})
        nodes.remove(node)
        self.assertTrue({node.name() for node in nodes} == {'x1', 'x2', 'out'})

    def test_retrieve_node(self):
        if False:
            for i in range(10):
                print('nop')
        graph = build_graph()
        nodes = []
        for i in range(len(graph.nodes())):
            nodes.append(graph.retrieve_node(i))
        for node in nodes:
            self.assertTrue(node in graph.nodes())

    def resolve_hazard(self):
        if False:
            return 10
        pass

def build_graph():
    if False:
        return 10
    prog = base.core.ProgramDesc()
    block = prog.block(0)
    shape = [10, 20]
    x1 = block.var(b'x1')
    x1.set_type(base.core.VarDesc.VarType.LOD_TENSOR)
    x1.set_shape(shape)
    x2 = block.var(b'x2')
    x2.set_type(base.core.VarDesc.VarType.LOD_TENSOR)
    x2.set_shape(shape)
    out = block.var(b'out')
    out.set_type(base.core.VarDesc.VarType.LOD_TENSOR)
    sum_op_desc = block.append_op()
    sum_op_desc.set_type('sum')
    sum_op_desc.set_input('X', ['x1', 'x2'])
    sum_op_desc.set_output('Out', ['out'])
    sum_op_desc.check_attrs()
    sum_op_desc.infer_shape(block)
    graph = base.core.Graph(prog)
    return graph
if __name__ == '__main__':
    unittest.main()