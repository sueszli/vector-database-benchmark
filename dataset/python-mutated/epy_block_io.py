import inspect
import collections
TYPE_MAP = {'complex64': 'complex', 'complex': 'complex', 'float32': 'float', 'float': 'float', 'int32': 'int', 'uint32': 'int', 'int16': 'short', 'uint16': 'short', 'int8': 'byte', 'uint8': 'byte'}
BlockIO = collections.namedtuple('BlockIO', 'name cls params sinks sources doc callbacks')

def _ports(sigs, msgs):
    if False:
        while True:
            i = 10
    ports = list()
    for (i, dtype) in enumerate(sigs):
        port_type = TYPE_MAP.get(dtype.base.name, None)
        if not port_type:
            raise ValueError("Can't map {0!r} to GRC port type".format(dtype))
        vlen = dtype.shape[0] if len(dtype.shape) > 0 else 1
        ports.append((str(i), port_type, vlen))
    for msg_key in msgs:
        if msg_key == 'system':
            continue
        ports.append((msg_key, 'message', 1))
    return ports

def _find_block_class(source_code, cls):
    if False:
        i = 10
        return i + 15
    ns = {}
    try:
        exec(source_code, ns)
    except Exception as e:
        raise ValueError("Can't interpret source code: " + str(e))
    for var in ns.values():
        if inspect.isclass(var) and issubclass(var, cls):
            return var
    raise ValueError('No python block class found in code')

def extract(cls):
    if False:
        i = 10
        return i + 15
    try:
        from gnuradio import gr
        import pmt
    except ImportError:
        raise EnvironmentError("Can't import GNU Radio")
    if not inspect.isclass(cls):
        cls = _find_block_class(cls, gr.gateway.gateway_block)
    spec = inspect.getfullargspec(cls.__init__)
    init_args = spec.args[1:]
    defaults = [repr(arg) for arg in spec.defaults or ()]
    doc = cls.__doc__ or cls.__init__.__doc__ or ''
    cls_name = cls.__name__
    if len(defaults) + 1 != len(spec.args):
        raise ValueError('Need all __init__ arguments to have default values')
    try:
        instance = cls()
    except Exception as e:
        raise RuntimeError("Can't create an instance of your block: " + str(e))
    name = instance.name()
    params = list(zip(init_args, defaults))

    def settable(attr):
        if False:
            for i in range(10):
                print('nop')
        try:
            return callable(getattr(cls, attr).fset)
        except AttributeError:
            return attr in instance.__dict__
    callbacks = [attr for attr in dir(instance) if attr in init_args and settable(attr)]
    sinks = _ports(instance.in_sig(), pmt.to_python(instance.message_ports_in()))
    sources = _ports(instance.out_sig(), pmt.to_python(instance.message_ports_out()))
    return BlockIO(name, cls_name, params, sinks, sources, doc, callbacks)
if __name__ == '__main__':
    blk_code = '\nimport numpy as np\nfrom gnuradio import gr\nimport pmt\n\nclass blk(gr.sync_block):\n    def __init__(self, param1=None, param2=None, param3=None):\n        "Test Docu"\n        gr.sync_block.__init__(\n            self,\n            name=\'Embedded Python Block\',\n            in_sig = (np.float32,),\n            out_sig = (np.float32,np.complex64,),\n        )\n        self.message_port_register_in(pmt.intern(\'msg_in\'))\n        self.message_port_register_out(pmt.intern(\'msg_out\'))\n        self.param1 = param1\n        self._param2 = param2\n        self._param3 = param3\n\n    @property\n    def param2(self):\n        return self._param2\n\n    @property\n    def param3(self):\n        return self._param3\n\n    @param3.setter\n    def param3(self, value):\n        self._param3 = value\n\n    def work(self, inputs_items, output_items):\n        return 10\n    '
    from pprint import pprint
    pprint(dict(extract(blk_code)._asdict()))