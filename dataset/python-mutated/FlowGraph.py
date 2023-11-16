import collections
import itertools
import sys
import types
import logging
from operator import methodcaller, attrgetter
from . import Messages, blocks
from .Constants import FLOW_GRAPH_FILE_FORMAT_VERSION
from .base import Element
from .utils import expr_utils
from .utils.backports import shlex
log = logging.getLogger(__name__)

class FlowGraph(Element):
    is_flow_graph = True

    def __init__(self, parent):
        if False:
            return 10
        '\n        Make a flow graph from the arguments.\n\n        Args:\n            parent: a platforms with blocks and element factories\n\n        Returns:\n            the flow graph object\n        '
        Element.__init__(self, parent)
        self.options_block = self.parent_platform.make_block(self, 'options')
        self.blocks = [self.options_block]
        self.connections = set()
        self._eval_cache = {}
        self.namespace = {}
        self.imported_names = []
        self.grc_file_path = ''

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'FlowGraph - {}({})'.format(self.get_option('title'), self.get_option('id'))

    def imports(self):
        if False:
            while True:
                i = 10
        '\n        Get a set of all import statements (Python) in this flow graph namespace.\n\n        Returns:\n            a list of import statements\n        '
        return [block.templates.render('imports') for block in self.iter_enabled_blocks()]

    def get_variables(self):
        if False:
            while True:
                i = 10
        '\n        Get a list of all variables (Python) in this flow graph namespace.\n        Exclude parameterized variables.\n\n        Returns:\n            a sorted list of variable blocks in order of dependency (indep -> dep)\n        '
        variables = [block for block in self.iter_enabled_blocks() if block.is_variable]
        return expr_utils.sort_objects(variables, attrgetter('name'), methodcaller('get_var_make'))

    def get_parameters(self):
        if False:
            i = 10
            return i + 15
        '\n        Get a list of all parameterized variables in this flow graph namespace.\n\n        Returns:\n            a list of parameterized variables\n        '
        parameters = [b for b in self.iter_enabled_blocks() if b.key == 'parameter']
        return parameters

    def get_snippets(self):
        if False:
            print('Hello World!')
        '\n        Get a set of all code snippets (Python) in this flow graph namespace.\n\n        Returns:\n            a list of code snippets\n        '
        return [b for b in self.iter_enabled_blocks() if b.key == 'snippet']

    def get_snippets_dict(self, section=None):
        if False:
            i = 10
            return i + 15
        '\n        Get a dictionary of code snippet information for a particular section.\n\n        Args:\n            section: string specifier of section of snippets to return, section=None returns all\n\n        Returns:\n            a list of code snippets dicts\n        '
        snippets = self.get_snippets()
        if not snippets:
            return []
        output = []
        for snip in snippets:
            d = {}
            sect = snip.params['section'].value
            d['section'] = sect
            d['priority'] = snip.params['priority'].value
            d['lines'] = snip.params['code'].value.splitlines()
            d['def'] = 'def snipfcn_{}(self):'.format(snip.name)
            d['call'] = 'snipfcn_{}(tb)'.format(snip.name)
            if not len(d['lines']):
                Messages.send_warning('Ignoring empty snippet from canvas')
            elif not section or sect == section:
                output.append(d)
        if section:
            output = sorted(output, key=lambda x: x['priority'], reverse=True)
        return output

    def get_monitors(self):
        if False:
            i = 10
            return i + 15
        '\n        Get a list of all ControlPort monitors\n        '
        monitors = [b for b in self.iter_enabled_blocks() if 'ctrlport_monitor' in b.key]
        return monitors

    def get_python_modules(self):
        if False:
            while True:
                i = 10
        'Iterate over custom code block ID and Source'
        for block in self.iter_enabled_blocks():
            if block.key == 'epy_module':
                yield (block.name, block.params['source_code'].get_value())

    def iter_enabled_blocks(self):
        if False:
            print('Hello World!')
        '\n        Get an iterator of all blocks that are enabled and not bypassed.\n        '
        return (block for block in self.blocks if block.enabled)

    def get_enabled_blocks(self):
        if False:
            return 10
        '\n        Get a list of all blocks that are enabled and not bypassed.\n\n        Returns:\n            a list of blocks\n        '
        return list(self.iter_enabled_blocks())

    def get_bypassed_blocks(self):
        if False:
            print('Hello World!')
        '\n        Get a list of all blocks that are bypassed.\n\n        Returns:\n            a list of blocks\n        '
        return [block for block in self.blocks if block.get_bypassed()]

    def get_enabled_connections(self):
        if False:
            print('Hello World!')
        '\n        Get a list of all connections that are enabled.\n\n        Returns:\n            a list of connections\n        '
        return [connection for connection in self.connections if connection.enabled]

    def get_option(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the option for a given key.\n        The option comes from the special options block.\n\n        Args:\n            key: the param key for the options block\n\n        Returns:\n            the value held by that param\n        '
        return self.options_block.params[key].get_evaluated()

    def get_run_command(self, file_path, split=False):
        if False:
            while True:
                i = 10
        run_command = self.get_option('run_command')
        try:
            run_command = run_command.format(python=shlex.quote(sys.executable), filename=shlex.quote(file_path))
            return shlex.split(run_command) if split else run_command
        except Exception as e:
            raise ValueError("Can't parse run command {!r}: {}".format(run_command, e))

    def get_imported_names(self):
        if False:
            while True:
                i = 10
        "\n        Get a lis of imported names.\n        These names may not be used as id's\n\n        Returns:\n            a list of imported names\n        "
        return self.imported_names

    def get_block(self, name):
        if False:
            i = 10
            return i + 15
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError('No block with name {!r}'.format(name))

    def get_elements(self):
        if False:
            for i in range(10):
                print('nop')
        elements = list(self.blocks)
        elements.extend(self.connections)
        return elements

    def children(self):
        if False:
            return 10
        return itertools.chain(self.blocks, self.connections)

    def rewrite(self):
        if False:
            return 10
        '\n        Flag the namespace to be renewed.\n        '
        self.renew_namespace()
        Element.rewrite(self)

    def renew_namespace(self):
        if False:
            i = 10
            return i + 15
        namespace = {}
        self.namespace.clear()
        for expr in self.imports():
            try:
                exec(expr, namespace)
            except ImportError:
                pass
            except Exception:
                log.exception('Failed to evaluate import expression "{0}"'.format(expr), exc_info=True)
                pass
        self.imported_names = list(namespace.keys())
        for (id, expr) in self.get_python_modules():
            try:
                module = types.ModuleType(id)
                exec(expr, module.__dict__)
                namespace[id] = module
            except Exception:
                log.exception('Failed to evaluate expression in module {0}'.format(id), exc_info=True)
                pass
        np = {}
        for parameter_block in self.get_parameters():
            try:
                value = eval(parameter_block.params['value'].to_code(), namespace)
                np[parameter_block.name] = value
            except Exception:
                log.exception('Failed to evaluate parameter block {0}'.format(parameter_block.name), exc_info=True)
                pass
        namespace.update(np)
        self.namespace.update(namespace)
        for variable_block in self.get_variables():
            try:
                variable_block.rewrite()
                value = eval(variable_block.value, namespace, variable_block.namespace)
                namespace[variable_block.name] = value
                self.namespace.update(namespace)
            except TypeError:
                pass
            except Exception:
                log.exception('Failed to evaluate variable block {0}'.format(variable_block.name), exc_info=True)
                pass
        self._eval_cache.clear()

    def evaluate(self, expr, namespace=None, local_namespace=None):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate the expression.\n        '
        if not expr:
            raise Exception('Cannot evaluate empty statement.')
        if namespace is not None:
            return eval(expr, namespace, local_namespace)
        else:
            return self._eval_cache.setdefault(expr, eval(expr, self.namespace, local_namespace))

    def new_block(self, block_id, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Get a new block of the specified key.\n        Add the block to the list of elements.\n\n        Args:\n            block_id: the block key\n\n        Returns:\n            the new block or None if not found\n        '
        if block_id == 'options':
            return self.options_block
        try:
            block = self.parent_platform.make_block(self, block_id, **kwargs)
            self.blocks.append(block)
        except KeyError:
            block = None
        return block

    def connect(self, porta, portb, params=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a connection between porta and portb.\n\n        Args:\n            porta: a port\n            portb: another port\n        @throw Exception bad connection\n\n        Returns:\n            the new connection\n        '
        connection = self.parent_platform.Connection(parent=self, source=porta, sink=portb)
        if params:
            connection.import_data(params)
        self.connections.add(connection)
        return connection

    def disconnect(self, *ports):
        if False:
            for i in range(10):
                print('nop')
        to_be_removed = [con for con in self.connections if any((port in con for port in ports))]
        for con in to_be_removed:
            self.remove_element(con)

    def remove_element(self, element):
        if False:
            i = 10
            return i + 15
        '\n        Remove the element from the list of elements.\n        If the element is a port, remove the whole block.\n        If the element is a block, remove its connections.\n        If the element is a connection, just remove the connection.\n        '
        if element is self.options_block:
            return
        if element.is_port:
            element = element.parent_block
        if element in self.blocks:
            self.disconnect(*element.ports())
            self.blocks.remove(element)
        elif element in self.connections:
            self.connections.remove(element)

    def export_data(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Export this flow graph to nested data.\n        Export all block and connection data.\n\n        Returns:\n            a nested data odict\n        '

        def block_order(b):
            if False:
                print('Hello World!')
            return (not b.is_variable, b.name)

        def get_file_format_version(data):
            if False:
                print('Hello World!')
            'Determine file format version based on available data'
            if any((isinstance(c, dict) for c in data['connections'])):
                return 2
            return 1

        def sort_connection_key(connection_info):
            if False:
                print('Hello World!')
            if isinstance(connection_info, dict):
                return [connection_info.get('src_blk_id'), connection_info.get('src_port_id'), connection_info.get('snk_blk_id'), connection_info.get('snk_port_id')]
            return connection_info
        data = collections.OrderedDict()
        data['options'] = self.options_block.export_data()
        data['blocks'] = [b.export_data() for b in sorted(self.blocks, key=block_order) if b is not self.options_block]
        data['connections'] = sorted((c.export_data() for c in self.connections), key=sort_connection_key)
        data['metadata'] = {'file_format': get_file_format_version(data), 'grc_version': self.parent_platform.config.version}
        return data

    def _build_depending_hier_block(self, block_id):
        if False:
            for i in range(10):
                print('nop')
        path_param = self.options_block.params['hier_block_src_path']
        file_path = self.parent_platform.find_file_in_paths(filename=block_id + '.grc', paths=path_param.get_value(), cwd=self.grc_file_path)
        if file_path:
            self.parent_platform.load_and_generate_flow_graph(file_path, hier_only=True)
            return self.new_block(block_id)

    def import_data(self, data):
        if False:
            print('Hello World!')
        '\n        Import blocks and connections into this flow graph.\n        Clear this flow graph of all previous blocks and connections.\n        Any blocks or connections in error will be ignored.\n\n        Args:\n            data: the nested data odict\n        '
        del self.blocks[:]
        self.connections.clear()
        file_format = data['metadata']['file_format']
        self.options_block.import_data(name='', **data.get('options', {}))
        self.blocks.append(self.options_block)
        for block_data in data.get('blocks', []):
            block_id = block_data['id']
            block = self.new_block(block_id) or self._build_depending_hier_block(block_id) or self.new_block(block_id='_dummy', missing_block_id=block_id, **block_data)
            block.import_data(**block_data)
        self.rewrite()

        def verify_and_get_port(key, block, dir):
            if False:
                for i in range(10):
                    print('nop')
            ports = block.sinks if dir == 'sink' else block.sources
            for port in ports:
                if key == port.key or key + '0' == port.key:
                    break
                if not key.isdigit() and port.dtype == '' and (key == port.name):
                    break
            else:
                if block.is_dummy_block:
                    port = block.add_missing_port(key, dir)
                else:
                    raise LookupError('%s key %r not in %s block keys' % (dir, key, dir))
            return port
        had_connect_errors = False
        _blocks = {block.name: block for block in self.blocks}
        for connection_info in data.get('connections', []):
            if isinstance(connection_info, (list, tuple)) and len(connection_info) == 4:
                (src_blk_id, src_port_id, snk_blk_id, snk_port_id) = connection_info
                conn_params = {}
            elif isinstance(connection_info, dict):
                src_blk_id = connection_info.get('src_blk_id')
                src_port_id = connection_info.get('src_port_id')
                snk_blk_id = connection_info.get('snk_blk_id')
                snk_port_id = connection_info.get('snk_port_id')
                conn_params = connection_info.get('params', {})
            else:
                Messages.send_error_load(f'Invalid connection format detected!')
                had_connect_errors = True
                continue
            try:
                source_block = _blocks[src_blk_id]
                sink_block = _blocks[snk_blk_id]
                if file_format < 1:
                    (src_port_id, snk_port_id) = _update_old_message_port_keys(src_port_id, snk_port_id, source_block, sink_block)
                source_port = verify_and_get_port(src_port_id, source_block, 'source')
                sink_port = verify_and_get_port(snk_port_id, sink_block, 'sink')
                self.connect(source_port, sink_port, conn_params)
            except (KeyError, LookupError) as e:
                Messages.send_error_load('Connection between {}({}) and {}({}) could not be made.\n\t{}'.format(src_blk_id, src_port_id, snk_blk_id, snk_port_id, e))
                had_connect_errors = True
        for block in self.blocks:
            if block.is_dummy_block:
                block.rewrite()
                block.add_error_message('Block id "{}" not found.'.format(block.key))
        self.rewrite()
        return had_connect_errors

def _update_old_message_port_keys(source_key, sink_key, source_block, sink_block):
    if False:
        for i in range(10):
            print('nop')
    "\n    Backward compatibility for message port keys\n\n    Message ports use their names as key (like in the 'connect' method).\n    Flowgraph files from former versions still have numeric keys stored for\n    message connections. These have to be replaced by the name of the\n    respective port. The correct message port is deduced from the integer\n    value of the key (assuming the order has not changed).\n\n    The connection ends are updated only if both ends translate into a\n    message port.\n    "
    try:
        source_port = source_block.sources[int(source_key)]
        sink_port = sink_block.sinks[int(sink_key)]
        if source_port.dtype == 'message' and sink_port.dtype == 'message':
            (source_key, sink_key) = (source_port.key, sink_port.key)
    except (ValueError, IndexError):
        pass
    return (source_key, sink_key)