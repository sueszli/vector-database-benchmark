from torch.testing._internal.opinfo.core import BinaryUfuncInfo, OpInfo, ReductionOpInfo, UnaryUfuncInfo

def _find_referenced_opinfo(referenced_name, variant_name, *, op_db=None):
    if False:
        return 10
    '\n    Finds the OpInfo with the given name that has no variant name.\n    '
    if op_db is None:
        from torch.testing._internal.common_methods_invocations import op_db
    for opinfo in op_db:
        if opinfo.name == referenced_name and opinfo.variant_test_name == variant_name:
            return opinfo

def _inherit_constructor_args(name, op, inherited, overrides):
    if False:
        return 10
    common_kwargs = {'name': name, 'op': op, 'aliases': None, 'method_variant': None, 'inplace_variant': None, 'supports_scripting': False}
    kwargs = inherited.copy()
    if 'kwargs' in kwargs:
        kwargs.update(kwargs['kwargs'])
        del kwargs['kwargs']
    if 'self' in kwargs:
        del kwargs['self']
    if '__class__' in kwargs:
        del kwargs['__class__']
    if 'skips' in kwargs:
        del kwargs['skips']
    if 'decorators' in kwargs:
        del kwargs['decorators']
    kwargs.update(common_kwargs)
    kwargs.update(overrides)
    kwargs['supports_autograd'] = False
    kwargs['supports_gradgrad'] = False
    kwargs['supports_fwgrad_bwgrad'] = False
    kwargs['supports_inplace_autograd'] = False
    kwargs['supports_forward_ad'] = False
    return kwargs

class PythonRefInfo(OpInfo):
    """
    An OpInfo for a Python reference of an OpInfo base class operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', validate_view_consistency=True, **kwargs):
        if False:
            print('Hello World!')
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        self.validate_view_consistency = validate_view_consistency
        assert isinstance(self.torch_opinfo, OpInfo)
        inherited = self.torch_opinfo._original_opinfo_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)

class ReductionPythonRefInfo(ReductionOpInfo):
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', **kwargs):
        if False:
            print('Hello World!')
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        assert isinstance(self.torch_opinfo, ReductionOpInfo)
        inherited = self.torch_opinfo._original_reduction_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        self.validate_view_consistency = False
        super().__init__(**ukwargs)

class ElementwiseUnaryPythonRefInfo(UnaryUfuncInfo):
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', validate_view_consistency=True, **kwargs):
        if False:
            i = 10
            return i + 15
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        self.validate_view_consistency = validate_view_consistency
        assert isinstance(self.torch_opinfo, UnaryUfuncInfo)
        inherited = self.torch_opinfo._original_unary_ufunc_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)

class ElementwiseBinaryPythonRefInfo(BinaryUfuncInfo):
    """
    An OpInfo for a Python reference of an elementwise binary operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', **kwargs):
        if False:
            i = 10
            return i + 15
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        assert isinstance(self.torch_opinfo, BinaryUfuncInfo)
        inherited = self.torch_opinfo._original_binary_ufunc_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)