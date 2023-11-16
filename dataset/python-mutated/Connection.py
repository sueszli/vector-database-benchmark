"""
Copyright 2008-2015 Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
import collections
from .base import Element
from .Constants import ALIASES_OF
from .utils.descriptors import lazy_property

class Connection(Element):
    """
    Stores information about a connection between two block ports. This class
    knows:
    - Where the source and sink ports are (on which blocks)
    - The domain (message, stream, ...)
    - Which parameters are associated with this connection
    """
    is_connection = True
    documentation = {'': ''}

    def __init__(self, parent, source, sink):
        if False:
            i = 10
            return i + 15
        '\n        Make a new connection given the parent and 2 ports.\n\n        Args:\n            parent: the parent of this element (a flow graph)\n            source: a port (any direction)\n            sink: a port (any direction)\n        @throws Error cannot make connection\n\n        Returns:\n            a new connection\n        '
        Element.__init__(self, parent)
        if not source.is_source:
            (source, sink) = (sink, source)
        if not source.is_source:
            raise ValueError('Connection could not isolate source')
        if not sink.is_sink:
            raise ValueError('Connection could not isolate sink')
        self.source_port = source
        self.sink_port = sink
        param_factory = self.parent_platform.make_param
        conn_parameters = self.parent_platform.connection_params.get(self.type, {})
        self.params = collections.OrderedDict(((data['id'], param_factory(parent=self, **data)) for data in conn_parameters))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'Connection (\n\t{}\n\t\t{}\n\t{}\n\t\t{}\n)'.format(self.source_block, self.source_port, self.sink_block, self.sink_port)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.source_port == other.source_port and self.sink_port == other.sink_port

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.source_port, self.sink_port))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter((self.source_port, self.sink_port))

    def children(self):
        if False:
            return 10
        ' This includes the connection parameters '
        return self.params.values()

    @lazy_property
    def source_block(self):
        if False:
            for i in range(10):
                print('nop')
        return self.source_port.parent_block

    @lazy_property
    def sink_block(self):
        if False:
            i = 10
            return i + 15
        return self.sink_port.parent_block

    @lazy_property
    def type(self):
        if False:
            return 10
        return (self.source_port.domain, self.sink_port.domain)

    @property
    def enabled(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the enabled state of this connection.\n\n        Returns:\n            true if source and sink blocks are enabled\n        '
        return self.source_block.enabled and self.sink_block.enabled

    @property
    def label(self):
        if False:
            print('Hello World!')
        ' Returns a label for dialogs '
        (src_domain, sink_domain) = [self.parent_platform.domains[d].name for d in self.type]
        return f'Connection ({src_domain} → {sink_domain})'

    @property
    def namespace_templates(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns everything we want to have available in the template rendering'
        return {key: param.template_arg for (key, param) in self.params.items()}

    def validate(self):
        if False:
            print('Hello World!')
        '\n        Validate the connections.\n        The ports must match in io size.\n        '
        Element.validate(self)
        platform = self.parent_platform
        if self.type not in platform.connection_templates:
            self.add_error_message('No connection known between domains "{}" and "{}"'.format(*self.type))
        source_dtype = self.source_port.dtype
        sink_dtype = self.sink_port.dtype
        if source_dtype != sink_dtype and source_dtype not in ALIASES_OF.get(sink_dtype, set()):
            self.add_error_message('Source IO type "{}" does not match sink IO type "{}".'.format(source_dtype, sink_dtype))
        source_size = self.source_port.item_size
        sink_size = self.sink_port.item_size
        if source_size != sink_size:
            self.add_error_message('Source IO size "{}" does not match sink IO size "{}".'.format(source_size, sink_size))

    def export_data(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Export this connection's info.\n\n        Returns:\n            A tuple with connection info, and parameters.\n        "
        if self.params:
            return {'src_blk_id': self.source_block.name, 'src_port_id': self.source_port.key, 'snk_blk_id': self.sink_block.name, 'snk_port_id': self.sink_port.key, 'params': collections.OrderedDict(sorted(((param_id, param.value) for (param_id, param) in self.params.items())))}
        return [self.source_block.name, self.source_port.key, self.sink_block.name, self.sink_port.key]

    def import_data(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Import connection parameters.\n        '
        for (key, value) in params.items():
            try:
                self.params[key].set_value(value)
            except KeyError:
                continue