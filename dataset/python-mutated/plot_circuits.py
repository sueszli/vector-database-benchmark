"""
========
Circuits
========

Convert a Boolean circuit to an equivalent Boolean formula.

A Boolean circuit can be exponentially more expressive than an
equivalent formula in the worst case, since the circuit can reuse
subcircuits multiple times, whereas a formula cannot reuse subformulas
more than once. Thus creating a Boolean formula from a Boolean circuit
in this way may be infeasible if the circuit is large.

"""
import matplotlib.pyplot as plt
import networkx as nx

def circuit_to_formula(circuit):
    if False:
        for i in range(10):
            print('nop')
    formula = nx.dag_to_branching(circuit)
    for v in formula:
        source = formula.nodes[v]['source']
        formula.nodes[v]['label'] = circuit.nodes[source]['label']
    return formula

def formula_to_string(formula):
    if False:
        for i in range(10):
            print('nop')

    def _to_string(formula, root):
        if False:
            while True:
                i = 10
        label = formula.nodes[root]['label']
        if not formula[root]:
            return label
        children = formula[root]
        if len(children) == 1:
            child = nx.utils.arbitrary_element(children)
            return f'{label}({_to_string(formula, child)})'
        (left, right) = formula[root]
        left_subformula = _to_string(formula, left)
        right_subformula = _to_string(formula, right)
        return f'({left_subformula} {label} {right_subformula})'
    root = next((v for (v, d) in formula.in_degree() if d == 0))
    return _to_string(formula, root)
circuit = nx.DiGraph()
circuit.add_node(0, label='∧', layer=0)
circuit.add_node(1, label='∨', layer=1)
circuit.add_node(2, label='∨', layer=1)
circuit.add_edge(0, 1)
circuit.add_edge(0, 2)
circuit.add_node(3, label='x', layer=2)
circuit.add_node(4, label='y', layer=2)
circuit.add_node(5, label='¬', layer=2)
circuit.add_edge(1, 3)
circuit.add_edge(1, 4)
circuit.add_edge(2, 4)
circuit.add_edge(2, 5)
circuit.add_node(6, label='z', layer=3)
circuit.add_edge(5, 6)
formula = circuit_to_formula(circuit)
print(formula_to_string(formula))
labels = nx.get_node_attributes(circuit, 'label')
options = {'node_size': 600, 'alpha': 0.5, 'node_color': 'blue', 'labels': labels, 'font_size': 22}
plt.figure(figsize=(8, 8))
pos = nx.multipartite_layout(circuit, subset_key='layer')
nx.draw_networkx(circuit, pos, **options)
plt.title(formula_to_string(formula))
plt.axis('equal')
plt.show()