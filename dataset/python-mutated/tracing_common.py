"""Code shared between the async and the sync test_decorator files."""
from collections import defaultdict
import os
from azure.core.settings import settings
from azure.core.tracing.ext.opencensus_span import OpenCensusSpan
from opencensus.trace import execution_context
from opencensus.trace.base_exporter import Exporter
from opencensus.trace.span_data import SpanData
try:
    from unittest import mock
except ImportError:
    import mock

class ContextHelper(object):

    def __init__(self, environ=None, tracer_to_use=None):
        if False:
            for i in range(10):
                print('nop')
        if environ is None:
            environ = {}
        self.orig_tracer = OpenCensusSpan.get_current_tracer()
        self.orig_current_span = OpenCensusSpan.get_current_span()
        self.os_env = mock.patch.dict(os.environ, environ)
        self.tracer_to_use = tracer_to_use

    def __enter__(self):
        if False:
            print('Hello World!')
        self.orig_tracer = OpenCensusSpan.get_current_tracer()
        self.orig_current_span = OpenCensusSpan.get_current_span()
        execution_context.clear()
        if self.tracer_to_use is not None:
            settings.tracing_implementation.set_value(self.tracer_to_use)
        self.os_env.start()
        execution_context.clear()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        OpenCensusSpan.set_current_tracer(self.orig_tracer)
        OpenCensusSpan.set_current_span(self.orig_current_span)
        settings.tracing_implementation.unset_value()
        self.os_env.stop()

class Node:

    def __init__(self, span_data):
        if False:
            print('Hello World!')
        self.span_data = span_data
        self.parent = None
        self.children = []

class MockExporter(Exporter):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.root = None
        self._all_nodes = []
        self.parent_dict = defaultdict(list)

    def export(self, span_data):
        if False:
            return 10
        sp = span_data[0]
        node = Node(sp)
        if not node.span_data.parent_span_id:
            self.root = node
        parent_span_id = node.span_data.parent_span_id
        self.parent_dict[parent_span_id].append(node)
        self._all_nodes.append(node)

    def build_tree(self):
        if False:
            print('Hello World!')
        for node in self._all_nodes:
            if node.span_data.span_id in self.parent_dict:
                node.children = sorted(self.parent_dict[node.span_data.span_id], key=lambda x: x.span_data.start_time)