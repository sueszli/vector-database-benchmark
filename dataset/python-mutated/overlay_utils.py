"""Utilities for writing overlays."""
from pytype.abstract import abstract
from pytype.abstract import function
from pytype.pytd import pytd
from pytype.typegraph import cfg
PARAM_TYPES = (cfg.Variable, abstract.Class, abstract.TypeParameter, abstract.Union, abstract.Unsolvable)

class Param:
    """Internal representation of method parameters."""

    def __init__(self, name, typ=None, default=None):
        if False:
            i = 10
            return i + 15
        if typ:
            assert isinstance(typ, PARAM_TYPES), (typ, type(typ))
        self.name = name
        self.typ = typ
        self.default = default

    def unsolvable(self, ctx, node):
        if False:
            return 10
        'Replace None values for typ and default with unsolvable.'
        self.typ = self.typ or ctx.convert.unsolvable
        self.default = self.default or ctx.new_unsolvable(node)
        return self

    def __repr__(self):
        if False:
            return 10
        return f'Param({self.name}, {self.typ!r}, {self.default!r})'

def make_method(ctx, node, name, params=None, posonly_count=0, kwonly_params=None, return_type=None, self_param=None, varargs=None, kwargs=None, kind=pytd.MethodKind.METHOD):
    if False:
        while True:
            i = 10
    'Make a method from params.\n\n  Args:\n    ctx: The context\n    node: Node to create the method variable at\n    name: The method name\n    params: Positional params [type: [Param]]\n    posonly_count: Number of positional-only parameters\n    kwonly_params: Keyword only params [type: [Param]]\n    return_type: Return type [type: PARAM_TYPES]\n    self_param: Self param [type: Param, defaults to self: Any]\n    varargs: Varargs param [type: Param, allows *args to be named and typed]\n    kwargs: Kwargs param [type: Param, allows **kwargs to be named and typed]\n    kind: The method kind\n\n  Returns:\n    A new method wrapped in a variable.\n  '

    def _process_annotation(param):
        if False:
            for i in range(10):
                print('nop')
        'Process a single param into annotations.'
        param_type = param.typ
        if not param_type:
            return
        elif isinstance(param_type, cfg.Variable):
            types = param_type.data
            if len(types) == 1:
                annotations[param.name] = types[0].cls
            else:
                t = abstract.Union([t.cls for t in types], ctx)
                annotations[param.name] = t
        else:
            annotations[param.name] = param_type
    params = params or []
    kwonly_params = kwonly_params or []
    if kind in (pytd.MethodKind.METHOD, pytd.MethodKind.PROPERTY):
        self_param = [self_param or Param('self', None, None)]
    elif kind == pytd.MethodKind.CLASSMETHOD:
        self_param = [Param('cls', None, None)]
    else:
        assert kind == pytd.MethodKind.STATICMETHOD
        self_param = []
    annotations = {}
    params = self_param + params
    return_param = Param('return', return_type, None) if return_type else None
    special_params = [x for x in (return_param, varargs, kwargs) if x]
    for param in special_params + params + kwonly_params:
        _process_annotation(param)
    names = lambda xs: tuple((x.name for x in xs))
    param_names = names(params)
    kwonly_names = names(kwonly_params)
    defaults = {x.name: x.default for x in params + kwonly_params if x.default}
    varargs_name = varargs.name if varargs else None
    kwargs_name = kwargs.name if kwargs else None
    ret = abstract.SimpleFunction.build(name=name, param_names=param_names, posonly_count=posonly_count, varargs_name=varargs_name, kwonly_params=kwonly_names, kwargs_name=kwargs_name, defaults=defaults, annotations=annotations, ctx=ctx)
    ret.signature.check_defaults(ctx)
    retvar = ret.to_variable(node)
    if kind in (pytd.MethodKind.METHOD, pytd.MethodKind.PROPERTY):
        return retvar
    if kind == pytd.MethodKind.CLASSMETHOD:
        decorator = ctx.vm.load_special_builtin('classmethod')
    else:
        assert kind == pytd.MethodKind.STATICMETHOD
        decorator = ctx.vm.load_special_builtin('staticmethod')
    args = function.Args(posargs=(retvar,))
    return decorator.call(node, func=None, args=args)[1]

def add_base_class(node, cls, base_cls):
    if False:
        while True:
            i = 10
    'Inserts base_cls into the MRO of cls.'
    bases = cls.bases()
    base_cls_mro = {x.full_name for x in base_cls.mro}
    cls_bases = [x.data[0].full_name for x in bases]
    cls_mro = [x.full_name for x in cls.mro]
    bpos = [i for (i, x) in enumerate(cls_bases) if x in base_cls_mro]
    mpos = [i for (i, x) in enumerate(cls_mro) if x in base_cls_mro]
    if bpos:
        (bpos, mpos) = (bpos[0], mpos[0])
        bases.insert(bpos, base_cls.to_variable(node))
        cls.mro = cls.mro[:mpos] + (base_cls,) + cls.mro[mpos:]
    else:
        bases.append(base_cls.to_variable(node))
        cls.mro = cls.mro + (base_cls,)

def not_supported_yet(name, ctx, module, details=None):
    if False:
        for i in range(10):
            print('nop')
    pytd_type = ctx.loader.lookup_pytd(module, name)
    ctx.errorlog.not_supported_yet(ctx.vm.frames, pytd_type.name, details=details)
    if isinstance(pytd_type, pytd.Alias):
        type_to_convert = pytd_type.type
    else:
        type_to_convert = pytd_type
    try:
        return ctx.convert.constant_to_value(type_to_convert, node=ctx.root_node)
    except NotImplementedError:
        return ctx.convert.unsolvable