"""For generating Beam pipeline graph in DOT representation.

This module is experimental. No backwards-compatibility guarantees.
"""
import collections
import logging
import threading
from typing import DefaultDict
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union
import pydot
import apache_beam as beam
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import pipeline_instrument as inst
from apache_beam.runners.interactive.display import pipeline_graph_renderer

class PipelineGraph(object):
    """Creates a DOT representing the pipeline. Thread-safe. Runner agnostic."""

    def __init__(self, pipeline, default_vertex_attrs={'shape': 'box'}, default_edge_attrs=None, render_option=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor of PipelineGraph.\n\n    Examples:\n      graph = pipeline_graph.PipelineGraph(pipeline_proto)\n      graph.get_dot()\n\n      or\n\n      graph = pipeline_graph.PipelineGraph(pipeline)\n      graph.get_dot()\n\n    Args:\n      pipeline: (Pipeline proto) or (Pipeline) pipeline to be rendered.\n      default_vertex_attrs: (Dict[str, str]) a dict of default vertex attributes\n      default_edge_attrs: (Dict[str, str]) a dict of default edge attributes\n      render_option: (str) this parameter decides how the pipeline graph is\n          rendered. See display.pipeline_graph_renderer for available options.\n    '
        self._lock = threading.Lock()
        self._graph = None
        self._pipeline_instrument = None
        if isinstance(pipeline, beam.Pipeline):
            self._pipeline_instrument = inst.PipelineInstrument(pipeline, pipeline._options)
            self._pipeline_instrument.preprocess()
        if isinstance(pipeline, beam_runner_api_pb2.Pipeline):
            self._pipeline_proto = pipeline
        elif isinstance(pipeline, beam.Pipeline):
            self._pipeline_proto = pipeline.to_runner_api()
        else:
            raise TypeError('pipeline should either be a %s or %s, while %s is given' % (beam_runner_api_pb2.Pipeline, beam.Pipeline, type(pipeline)))
        self._consumers = collections.defaultdict(list)
        self._producers = {}
        for (transform_id, transform_proto) in self._top_level_transforms():
            for pcoll_id in transform_proto.inputs.values():
                self._consumers[pcoll_id].append(transform_id)
            for pcoll_id in transform_proto.outputs.values():
                self._producers[pcoll_id] = transform_id
        default_vertex_attrs = default_vertex_attrs or {'shape': 'box'}
        if 'color' not in default_vertex_attrs:
            default_vertex_attrs['color'] = 'blue'
        if 'fontcolor' not in default_vertex_attrs:
            default_vertex_attrs['fontcolor'] = 'blue'
        (vertex_dict, edge_dict) = self._generate_graph_dicts()
        self._construct_graph(vertex_dict, edge_dict, default_vertex_attrs, default_edge_attrs)
        self._renderer = pipeline_graph_renderer.get_renderer(render_option)

    def get_dot(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_graph().to_string()

    def display_graph(self):
        if False:
            return 10
        'Displays the graph generated.'
        rendered_graph = self._renderer.render_pipeline_graph(self)
        if ie.current_env().is_in_notebook:
            try:
                from IPython import display
                display.display(display.HTML(rendered_graph))
            except ImportError:
                logging.warning('Failed to import IPython display module when current environment is in a notebook. Cannot display the pipeline graph.')

    def _top_level_transforms(self):
        if False:
            return 10
        'Yields all top level PTransforms (subtransforms of the root PTransform).\n\n    Yields: (str, PTransform proto) ID, proto pair of top level PTransforms.\n    '
        transforms = self._pipeline_proto.components.transforms
        for root_transform_id in self._pipeline_proto.root_transform_ids:
            root_transform_proto = transforms[root_transform_id]
            for top_level_transform_id in root_transform_proto.subtransforms:
                top_level_transform_proto = transforms[top_level_transform_id]
                yield (top_level_transform_id, top_level_transform_proto)

    def _decorate(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Decorates label-ish values used for rendering in dot language.\n\n    Escapes special characters in the given str value for dot language. All\n    PTransform unique names are escaped implicitly in this module when building\n    dot representation. Otherwise, special characters will break the graph\n    rendered or cause runtime errors.\n    '
        return '"{}"'.format(value.replace('\\', '\\\\').replace('"', '\\"'))

    def _generate_graph_dicts(self):
        if False:
            print('Hello World!')
        "From pipeline_proto and other info, generate the graph.\n\n    Returns:\n      vertex_dict: (Dict[str, Dict[str, str]]) vertex mapped to attributes.\n      edge_dict: (Dict[(str, str), Dict[str, str]]) vertex pair mapped to the\n          edge's attribute.\n    "
        transforms = self._pipeline_proto.components.transforms
        vertex_dict = collections.defaultdict(dict)
        edge_dict = collections.defaultdict(dict)
        self._edge_to_vertex_pairs = collections.defaultdict(list)
        for (_, transform) in self._top_level_transforms():
            vertex_dict[self._decorate(transform.unique_name)] = {}
            for pcoll_id in transform.outputs.values():
                pcoll_node = None
                if self._pipeline_instrument:
                    cacheable = self._pipeline_instrument.cacheables.get(pcoll_id)
                    pcoll_node = cacheable.var if cacheable else None
                if not pcoll_node:
                    pcoll_node = 'pcoll%s' % (hash(pcoll_id) % 10000)
                    vertex_dict[pcoll_node] = {'shape': 'circle', 'label': ''}
                else:
                    vertex_dict[pcoll_node] = {'shape': 'circle'}
                if pcoll_id not in self._consumers:
                    self._edge_to_vertex_pairs[pcoll_id].append((self._decorate(transform.unique_name), pcoll_node))
                    edge_dict[self._decorate(transform.unique_name), pcoll_node] = {}
                else:
                    for consumer in self._consumers[pcoll_id]:
                        producer_name = self._decorate(transform.unique_name)
                        consumer_name = self._decorate(transforms[consumer].unique_name)
                        self._edge_to_vertex_pairs[pcoll_id].append((producer_name, pcoll_node))
                        edge_dict[producer_name, pcoll_node] = {}
                        self._edge_to_vertex_pairs[pcoll_id].append((pcoll_node, consumer_name))
                        edge_dict[pcoll_node, consumer_name] = {}
        return (vertex_dict, edge_dict)

    def _get_graph(self):
        if False:
            return 10
        'Returns pydot.Dot object for the pipeline graph.\n\n    The purpose of this method is to avoid accessing the graph while it is\n    updated. No one except for this method should be accessing _graph directly.\n\n    Returns:\n      (pydot.Dot)\n    '
        with self._lock:
            return self._graph

    def _construct_graph(self, vertex_dict, edge_dict, default_vertex_attrs, default_edge_attrs):
        if False:
            print('Hello World!')
        'Constructs the pydot.Dot object for the pipeline graph.\n\n    Args:\n      vertex_dict: (Dict[str, Dict[str, str]]) maps vertex names to attributes\n      edge_dict: (Dict[(str, str), Dict[str, str]]) maps vertex name pairs to\n          attributes\n      default_vertex_attrs: (Dict[str, str]) a dict of attributes\n      default_edge_attrs: (Dict[str, str]) a dict of attributes\n    '
        with self._lock:
            self._graph = pydot.Dot()
            if default_vertex_attrs:
                self._graph.set_node_defaults(**default_vertex_attrs)
            if default_edge_attrs:
                self._graph.set_edge_defaults(**default_edge_attrs)
            self._vertex_refs = {}
            self._edge_refs = {}
            for (vertex, vertex_attrs) in vertex_dict.items():
                vertex_ref = pydot.Node(vertex, **vertex_attrs)
                self._vertex_refs[vertex] = vertex_ref
                self._graph.add_node(vertex_ref)
            for (edge, edge_attrs) in edge_dict.items():
                vertex_src = self._vertex_refs[edge[0]]
                vertex_dst = self._vertex_refs[edge[1]]
                edge_ref = pydot.Edge(vertex_src, vertex_dst, **edge_attrs)
                self._edge_refs[edge] = edge_ref
                self._graph.add_edge(edge_ref)

    def _update_graph(self, vertex_dict=None, edge_dict=None):
        if False:
            i = 10
            return i + 15
        'Updates the pydot.Dot object with the given attribute update\n\n    Args:\n      vertex_dict: (Dict[str, Dict[str, str]]) maps vertex names to attributes\n      edge_dict: This should be\n          Either (Dict[str, Dict[str, str]]) which maps edge names to attributes\n          Or (Dict[(str, str), Dict[str, str]]) which maps vertex pairs to edge\n          attributes\n    '

        def set_attrs(ref, attrs):
            if False:
                print('Hello World!')
            for (attr_name, attr_val) in attrs.items():
                ref.set(attr_name, attr_val)
        with self._lock:
            if vertex_dict:
                for (vertex, vertex_attrs) in vertex_dict.items():
                    set_attrs(self._vertex_refs[vertex], vertex_attrs)
            if edge_dict:
                for (edge, edge_attrs) in edge_dict.items():
                    if isinstance(edge, tuple):
                        set_attrs(self._edge_refs[edge], edge_attrs)
                    else:
                        for vertex_pair in self._edge_to_vertex_pairs[edge]:
                            set_attrs(self._edge_refs[vertex_pair], edge_attrs)