from ast import literal_eval
from textwrap import dedent
from . import Block, register_build_in
from ._templates import MakoTemplates
from ._flags import Flags
from .. import utils
from ..base import Element
from ._build import build_params
DEFAULT_CODE = '"""\nEmbedded Python Blocks:\n\nEach time this file is saved, GRC will instantiate the first class it finds\nto get ports and parameters of your block. The arguments to __init__  will\nbe the parameters. All of them are required to have default values!\n"""\n\nimport numpy as np\nfrom gnuradio import gr\n\n\nclass blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block\n    """Embedded Python Block example - a simple multiply const"""\n\n    def __init__(self, example_param=1.0):  # only default arguments here\n        """arguments to this function show up as parameters in GRC"""\n        gr.sync_block.__init__(\n            self,\n            name=\'Embedded Python Block\',   # will show up in GRC\n            in_sig=[np.complex64],\n            out_sig=[np.complex64]\n        )\n        # if an attribute with the same name as a parameter is found,\n        # a callback is registered (properties work, too).\n        self.example_param = example_param\n\n    def work(self, input_items, output_items):\n        """example: multiply with constant"""\n        output_items[0][:] = input_items[0] * self.example_param\n        return len(output_items[0])\n'
DOC = "\nThis block represents an arbitrary GNU Radio Python Block.\n\nIts source code can be accessed through the parameter 'Code' which opens your editor. Each time you save changes in the editor, GRC will update the block. This includes the number, names and defaults of the parameters, the ports (stream and message) and the block name and documentation.\n\nBlock Documentation:\n(will be replaced the docstring of your block class)\n"

@register_build_in
class EPyBlock(Block):
    key = 'epy_block'
    label = 'Python Block'
    exempt_from_id_validation = True
    documentation = {'': DOC}
    parameters_data = build_params(params_raw=[dict(label='Code', id='_source_code', dtype='_multiline_python_external', default=DEFAULT_CODE, hide='part')], have_inputs=True, have_outputs=True, flags=Block.flags, block_id=key)
    inputs_data = []
    outputs_data = []

    def __init__(self, flow_graph, **kwargs):
        if False:
            print('Hello World!')
        super(EPyBlock, self).__init__(flow_graph, **kwargs)
        self.states['_io_cache'] = ''
        self.module_name = self.name
        self._epy_source_hash = -1
        self._epy_reload_error = None

    def rewrite(self):
        if False:
            i = 10
            return i + 15
        Element.rewrite(self)
        param_src = self.params['_source_code']
        src = param_src.get_value()
        src_hash = hash((self.name, src))
        if src_hash == self._epy_source_hash:
            return
        try:
            blk_io = utils.epy_block_io.extract(src)
        except Exception as e:
            self._epy_reload_error = ValueError(str(e))
            try:
                blk_io_args = literal_eval(self.states['_io_cache'])
                if len(blk_io_args) == 6:
                    blk_io_args += ([],)
                blk_io = utils.epy_block_io.BlockIO(*blk_io_args)
            except Exception:
                return
        else:
            self._epy_reload_error = None
            self.states['_io_cache'] = repr(tuple(blk_io))
        self._epy_source_hash = src_hash
        self.label = blk_io.name or blk_io.cls
        self.documentation = {'': blk_io.doc}
        self.module_name = '{}_{}'.format(self.parent_flowgraph.get_option('id'), self.name)
        self.templates['imports'] = 'import {} as {}  # embedded python block'.format(self.module_name, self.name)
        self.templates['make'] = '{mod}.{cls}({args})'.format(mod=self.name, cls=blk_io.cls, args=', '.join(('{0}=${{ {0} }}'.format(key) for (key, _) in blk_io.params)))
        self.templates['callbacks'] = ['{0} = ${{ {0} }}'.format(attr) for attr in blk_io.callbacks]
        self._update_params(blk_io.params)
        self._update_ports('in', self.sinks, blk_io.sinks, 'sink')
        self._update_ports('out', self.sources, blk_io.sources, 'source')
        super(EPyBlock, self).rewrite()

    def _update_params(self, params_in_src):
        if False:
            for i in range(10):
                print('nop')
        param_factory = self.parent_platform.make_param
        params = {}
        for (key, value) in self.params.copy().items():
            if hasattr(value, '__epy_param__'):
                params[key] = value
                del self.params[key]
        for (id_, value) in params_in_src:
            try:
                param = params[id_]
                if param.default == param.value:
                    param.set_value(value)
                param.default = str(value)
            except KeyError:
                param = param_factory(parent=self, id=id_, dtype='raw', value=value, name=id_.replace('_', ' ').title())
                setattr(param, '__epy_param__', True)
            self.params[id_] = param

    def _update_ports(self, label, ports, port_specs, direction):
        if False:
            print('Hello World!')
        port_factory = self.parent_platform.make_port
        ports_to_remove = list(ports)
        iter_ports = iter(ports)
        ports_new = []
        port_current = next(iter_ports, None)
        for (key, port_type, vlen) in port_specs:
            reuse_port = port_current is not None and port_current.dtype == port_type and (port_current.vlen == vlen) and (key.isdigit() or port_current.key == key)
            if reuse_port:
                ports_to_remove.remove(port_current)
                (port, port_current) = (port_current, next(iter_ports, None))
            else:
                n = dict(name=label + str(key), dtype=port_type, id=key)
                if port_type == 'message':
                    n['name'] = key
                    n['optional'] = '1'
                if vlen > 1:
                    n['vlen'] = str(vlen)
                port = port_factory(self, direction=direction, **n)
            ports_new.append(port)
        del ports[:]
        ports.extend(ports_new)
        self.parent_flowgraph.disconnect(*ports_to_remove)

    def validate(self):
        if False:
            i = 10
            return i + 15
        super(EPyBlock, self).validate()
        if self._epy_reload_error:
            self.params['_source_code'].add_error_message(str(self._epy_reload_error))

@register_build_in
class EPyModule(Block):
    key = 'epy_module'
    label = 'Python Module'
    exempt_from_id_validation = True
    documentation = {'': dedent("\n        This block lets you embed a python module in your flowgraph.\n\n        Code you put in this module is accessible in other blocks using the ID of this\n        block. Example:\n\n        If you put\n\n            a = 2\n\n            def double(arg):\n                return 2 * arg\n\n        in a Python Module Block with the ID 'stuff' you can use code like\n\n            stuff.a  # evals to 2\n            stuff.double(3)  # evals to 6\n\n        to set parameters of other blocks in your flowgraph.\n    ")}
    flags = Flags(Flags.SHOW_ID)
    parameters_data = build_params(params_raw=[dict(label='Code', id='source_code', dtype='_multiline_python_external', default='# this module will be imported in the into your flowgraph', hide='part')], have_inputs=False, have_outputs=False, flags=flags, block_id=key)

    def __init__(self, flow_graph, **kwargs):
        if False:
            while True:
                i = 10
        super(EPyModule, self).__init__(flow_graph, **kwargs)
        self.module_name = self.name

    def rewrite(self):
        if False:
            return 10
        super(EPyModule, self).rewrite()
        self.module_name = '{}_{}'.format(self.parent_flowgraph.get_option('id'), self.name)
        self.templates['imports'] = 'import {} as {}  # embedded python module'.format(self.module_name, self.name)