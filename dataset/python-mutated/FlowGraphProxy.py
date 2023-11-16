from ..utils import expr_utils
from operator import methodcaller, attrgetter

class FlowGraphProxy(object):

    def __init__(self, fg):
        if False:
            return 10
        self.orignal_flowgraph = fg

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        return getattr(self.orignal_flowgraph, item)

    def get_hier_block_stream_io(self, direction):
        if False:
            print('Hello World!')
        "\n        Get a list of stream io signatures for this flow graph.\n\n        Args:\n            direction: a string of 'in' or 'out'\n\n        Returns:\n            a list of dicts with: type, label, vlen, size, optional\n        "
        return [p for p in self.get_hier_block_io(direction) if p['type'] != 'message']

    def get_hier_block_message_io(self, direction):
        if False:
            while True:
                i = 10
        "\n        Get a list of message io signatures for this flow graph.\n\n        Args:\n            direction: a string of 'in' or 'out'\n\n        Returns:\n            a list of dicts with: type, label, vlen, size, optional\n        "
        return [p for p in self.get_hier_block_io(direction) if p['type'] == 'message']

    def get_hier_block_io(self, direction):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get a list of io ports for this flow graph.\n\n        Args:\n            direction: a string of 'in' or 'out'\n\n        Returns:\n            a list of dicts with: type, label, vlen, size, optional\n        "
        pads = self.get_pad_sources() if direction in ('sink', 'in') else self.get_pad_sinks() if direction in ('source', 'out') else []
        ports = []
        for pad in pads:
            type_param = pad.params['type']
            master = {'label': str(pad.params['label'].get_evaluated()), 'type': str(pad.params['type'].get_evaluated()), 'vlen': str(pad.params['vlen'].get_value()), 'size': type_param.options.attributes[type_param.get_value()]['size'], 'cpp_size': type_param.options.attributes[type_param.get_value()]['cpp_size'], 'optional': bool(pad.params['optional'].get_evaluated())}
            num_ports = pad.params['num_streams'].get_evaluated()
            if num_ports > 1:
                for i in range(num_ports):
                    clone = master.copy()
                    clone['label'] += str(i)
                    ports.append(clone)
            else:
                ports.append(master)
        return ports

    def get_pad_sources(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a list of pad source blocks sorted by id order.\n\n        Returns:\n            a list of pad source blocks in this flow graph\n        '
        pads = [b for b in self.get_enabled_blocks() if b.key == 'pad_source']
        return sorted(pads, key=lambda x: x.name)

    def get_pad_sinks(self):
        if False:
            i = 10
            return i + 15
        '\n        Get a list of pad sink blocks sorted by id order.\n\n        Returns:\n            a list of pad sink blocks in this flow graph\n        '
        pads = [b for b in self.get_enabled_blocks() if b.key == 'pad_sink']
        return sorted(pads, key=lambda x: x.name)

    def get_pad_port_global_key(self, port):
        if False:
            return 10
        '\n        Get the key for a port of a pad source/sink to use in connect()\n        This takes into account that pad blocks may have multiple ports\n\n        Returns:\n            the key (str)\n        '
        key_offset = 0
        pads = self.get_pad_sources() if port.is_source else self.get_pad_sinks()
        for pad in pads:
            is_message_pad = pad.params['type'].get_evaluated() == 'message'
            if port.parent == pad:
                if is_message_pad:
                    key = pad.params['label'].get_value()
                else:
                    key = str(key_offset + int(port.key))
                return key
            elif not is_message_pad:
                key_offset += len(pad.sinks) + len(pad.sources)
        return -1

    def get_cpp_variables(self):
        if False:
            return 10
        '\n        Get a list of all variables (C++) in this flow graph namespace.\n        Exclude parameterized variables.\n\n        Returns:\n            a sorted list of variable blocks in order of dependency (indep -> dep)\n        '
        variables = [block for block in self.iter_enabled_blocks() if block.is_variable]
        return expr_utils.sort_objects(variables, attrgetter('name'), methodcaller('get_cpp_var_make'))

    def includes(self):
        if False:
            print('Hello World!')
        '\n        Get a set of all include statements (C++) in this flow graph namespace.\n\n        Returns:\n            a list of #include statements\n        '
        return [block.cpp_templates.render('includes') for block in self.iter_enabled_blocks() if not (block.is_virtual_sink() or block.is_virtual_source())]

    def links(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a set of all libraries to link against (C++) in this flow graph namespace.\n\n        Returns:\n            a list of GNU Radio modules\n        '
        return [block.cpp_templates.render('link') for block in self.iter_enabled_blocks() if not (block.is_virtual_sink() or block.is_virtual_source())]

    def packages(self):
        if False:
            return 10
        '\n        Get a set of all packages  to find (C++) ( especially for oot modules ) in this flow graph namespace.\n\n        Returns:\n            a list of required packages\n        '
        return [block.cpp_templates.render('packages') for block in self.iter_enabled_blocks() if not (block.is_virtual_sink() or block.is_virtual_source())]

def get_hier_block_io(flow_graph, direction, domain=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a list of io ports for this flow graph.\n\n    Returns a list of dicts with: type, label, vlen, size, optional\n    '
    pads = flow_graph.get_pad_sources() if direction in ('sink', 'in') else flow_graph.get_pad_sinks() if direction in ('source', 'out') else []
    ports = []
    for pad in pads:
        type_param = pad.params['type']
        master = {'label': str(pad.params['label'].get_evaluated()), 'type': str(pad.params['type'].get_evaluated()), 'vlen': str(pad.params['vlen'].get_value()), 'size': type_param.options.attributes[type_param.get_value()]['size'], 'optional': bool(pad.params['optional'].get_evaluated())}
        num_ports = pad.params['num_streams'].get_evaluated()
        if num_ports > 1:
            for i in range(num_ports):
                clone = master.copy()
                clone['label'] += str(i)
                ports.append(clone)
        else:
            ports.append(master)
    if domain is not None:
        ports = [p for p in ports if p.domain == domain]
    return ports