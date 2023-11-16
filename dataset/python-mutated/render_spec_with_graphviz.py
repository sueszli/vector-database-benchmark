"""Renders DRAGNN specs with Graphviz."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import pygraphviz
from dragnn.protos import spec_pb2

def _component_contents(component):
    if False:
        return 10
    'Generates the label on component boxes.\n\n  Args:\n    component: spec_pb2.ComponentSpec proto\n\n  Returns:\n    String label\n  '
    return '<\n  <B>{name}</B><BR />\n  {transition_name}<BR />\n  {network_name}<BR />\n  {num_actions_str}<BR />\n  hidden: {num_hidden}\n  >'.format(name=component.name, transition_name=component.transition_system.registered_name, network_name=component.network_unit.registered_name, num_actions_str='{} action{}'.format(component.num_actions, 's' if component.num_actions != 1 else ''), num_hidden=component.network_unit.parameters.get('hidden_layer_sizes', 'not specified'))

def _linked_feature_label(linked_feature):
    if False:
        i = 10
        return i + 15
    'Generates the label on edges between components.\n\n  Args:\n    linked_feature: spec_pb2.LinkedFeatureChannel proto\n\n  Returns:\n    String label\n  '
    return '<\n  <B>{name}</B><BR />\n  F={num_features} D={projected_dim}<BR />\n  {fml}<BR />\n  <U>{source_translator}</U><BR />\n  <I>{source_layer}</I>\n  >'.format(name=linked_feature.name, num_features=linked_feature.size, projected_dim=linked_feature.embedding_dim, fml=linked_feature.fml, source_translator=linked_feature.source_translator, source_layer=linked_feature.source_layer)

def master_spec_graph(master_spec):
    if False:
        for i in range(10):
            print('nop')
    'Constructs a master spec graph.\n\n  Args:\n    master_spec: MasterSpec proto.\n\n  Raises:\n    TypeError, if master_spec is not the right type. N.B. that this may be\n    raised if you import proto classes in non-standard ways (e.g. dynamically).\n\n  Returns:\n    SVG graph contents as a string.\n  '
    if not isinstance(master_spec, spec_pb2.MasterSpec):
        raise TypeError('master_spec_graph() expects a MasterSpec input.')
    graph = pygraphviz.AGraph(directed=True)
    graph.node_attr.update(shape='box', style='filled', fillcolor='white', fontname='roboto, helvetica, arial', fontsize=11)
    graph.edge_attr.update(fontname='roboto, helvetica, arial', fontsize=11)
    for component in master_spec.component:
        graph.add_node(component.name, label=_component_contents(component))
    for component in master_spec.component:
        for linked_feature in component.linked_feature:
            graph.add_edge(linked_feature.source_component, component.name, label=_linked_feature_label(linked_feature))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return graph.draw(format='svg', prog='dot')