from collections import namedtuple
from numba.core import types, ir
from numba.core.typing import signature
_CallableNode = namedtuple('BoundFunc', ['func', 'sig'])

class ParforLoweringBuilder:
    """Helper class for building Numba-IR and lowering for Parfor.
    """

    def __init__(self, lowerer, scope, loc):
        if False:
            for i in range(10):
                print('nop')
        self._lowerer = lowerer
        self._scope = scope
        self._loc = loc

    @property
    def _context(self):
        if False:
            while True:
                i = 10
        return self._lowerer.context

    @property
    def _typingctx(self):
        if False:
            return 10
        return self._context.typing_context

    @property
    def _typemap(self):
        if False:
            print('Hello World!')
        return self._lowerer.fndesc.typemap

    @property
    def _calltypes(self):
        if False:
            while True:
                i = 10
        return self._lowerer.fndesc.calltypes

    def bind_global_function(self, fobj, ftype, args, kws={}):
        if False:
            while True:
                i = 10
        'Binds a global function to a variable.\n\n        Parameters\n        ----------\n        fobj : object\n            The function to be bound.\n        ftype : types.Type\n        args : Sequence[types.Type]\n        kws : Mapping[str, types.Type]\n\n        Returns\n        -------\n        callable: _CallableNode\n        '
        loc = self._loc
        varname = f'{fobj.__name__}_func'
        gvname = f'{fobj.__name__}'
        func_sig = self._typingctx.resolve_function_type(ftype, args, kws)
        func_var = self.assign(rhs=ir.Global(gvname, fobj, loc=loc), typ=ftype, name=varname)
        return _CallableNode(func=func_var, sig=func_sig)

    def make_const_variable(self, cval, typ, name='pf_const') -> ir.Var:
        if False:
            i = 10
            return i + 15
        'Makes a constant variable\n\n        Parameters\n        ----------\n        cval : object\n            The constant value\n        typ : types.Type\n            type of the value\n        name : str\n            variable name to store to\n\n        Returns\n        -------\n        res : ir.Var\n        '
        return self.assign(rhs=ir.Const(cval, loc=self._loc), typ=typ, name=name)

    def make_tuple_variable(self, varlist, name='pf_tuple') -> ir.Var:
        if False:
            while True:
                i = 10
        'Makes a tuple variable\n\n        Parameters\n        ----------\n        varlist : Sequence[ir.Var]\n            Variables containing the values to be stored.\n        name : str\n            variable name to store to\n\n        Returns\n        -------\n        res : ir.Var\n        '
        loc = self._loc
        vartys = [self._typemap[x.name] for x in varlist]
        tupty = types.Tuple.from_types(vartys)
        return self.assign(rhs=ir.Expr.build_tuple(varlist, loc), typ=tupty, name=name)

    def assign(self, rhs, typ, name='pf_assign') -> ir.Var:
        if False:
            return 10
        'Assign a value to a new variable\n\n        Parameters\n        ----------\n        rhs : object\n            The value\n        typ : types.Type\n            type of the value\n        name : str\n            variable name to store to\n\n        Returns\n        -------\n        res : ir.Var\n        '
        loc = self._loc
        var = self._scope.redefine(name, loc)
        self._typemap[var.name] = typ
        assign = ir.Assign(rhs, var, loc)
        self._lowerer.lower_inst(assign)
        return var

    def assign_inplace(self, rhs, typ, name) -> ir.Var:
        if False:
            i = 10
            return i + 15
        'Assign a value to a new variable or inplace if it already exist\n\n        Parameters\n        ----------\n        rhs : object\n            The value\n        typ : types.Type\n            type of the value\n        name : str\n            variable name to store to\n\n        Returns\n        -------\n        res : ir.Var\n        '
        loc = self._loc
        var = ir.Var(self._scope, name, loc)
        assign = ir.Assign(rhs, var, loc)
        self._typemap.setdefault(var.name, typ)
        self._lowerer.lower_inst(assign)
        return var

    def call(self, callable_node, args, kws={}) -> ir.Expr:
        if False:
            return 10
        'Call a bound callable\n\n        Parameters\n        ----------\n        callable_node : _CallableNode\n            The callee\n        args : Sequence[ir.Var]\n        kws : Mapping[str, ir.Var]\n\n        Returns\n        -------\n        res : ir.Expr\n            The expression node for the return value of the call\n        '
        call = ir.Expr.call(callable_node.func, args, kws, loc=self._loc)
        self._calltypes[call] = callable_node.sig
        return call

    def setitem(self, obj, index, val) -> ir.SetItem:
        if False:
            print('Hello World!')
        'Makes a setitem call\n\n        Parameters\n        ----------\n        obj : ir.Var\n            the object being indexed\n        index : ir.Var\n            the index\n        val : ir.Var\n            the value to be stored\n\n        Returns\n        -------\n        res : ir.SetItem\n        '
        loc = self._loc
        tm = self._typemap
        setitem = ir.SetItem(obj, index, val, loc=loc)
        self._lowerer.fndesc.calltypes[setitem] = signature(types.none, tm[obj.name], tm[index.name], tm[val.name])
        self._lowerer.lower_inst(setitem)
        return setitem

    def getitem(self, obj, index, typ) -> ir.Expr:
        if False:
            for i in range(10):
                print('nop')
        'Makes a getitem call\n\n        Parameters\n        ----------\n        obj : ir.Var\n            the object being indexed\n        index : ir.Var\n            the index\n        val : ir.Var\n            the ty\n\n        Returns\n        -------\n        res : ir.Expr\n            the retrieved value\n        '
        tm = self._typemap
        getitem = ir.Expr.getitem(obj, index, loc=self._loc)
        self._lowerer.fndesc.calltypes[getitem] = signature(typ, tm[obj.name], tm[index.name])
        return getitem