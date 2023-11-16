from . import Nodes, ExprNodes, FusedNode, TreeFragment, Pipeline, ParseTreeTransforms, Naming, UtilNodes
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
numpy_int_types = ['NPY_BYTE', 'NPY_INT8', 'NPY_SHORT', 'NPY_INT16', 'NPY_INT', 'NPY_INT32', 'NPY_LONG', 'NPY_LONGLONG', 'NPY_INT64']
numpy_uint_types = [tp.replace('NPY_', 'NPY_U') for tp in numpy_int_types]
numpy_numeric_types = numpy_int_types + numpy_uint_types + ['NPY_FLOAT', 'NPY_FLOAT32', 'NPY_DOUBLE', 'NPY_FLOAT64', 'NPY_LONGDOUBLE']

def _get_type_constant(pos, type_):
    if False:
        while True:
            i = 10
    if type_.is_complex:
        if type_ == PyrexTypes.c_float_complex_type:
            return 'NPY_CFLOAT'
        elif type_ == PyrexTypes.c_double_complex_type:
            return 'NPY_CDOUBLE'
        elif type_ == PyrexTypes.c_longdouble_complex_type:
            return 'NPY_CLONGDOUBLE'
    elif type_.is_numeric:
        postfix = type_.empty_declaration_code().upper().replace(' ', '')
        typename = 'NPY_%s' % postfix
        if typename in numpy_numeric_types:
            return typename
    elif type_.is_pyobject:
        return 'NPY_OBJECT'
    error(pos, "Type '%s' cannot be used as a ufunc argument" % type_)

class _FindCFuncDefNode(TreeVisitor):
    """
    Finds the CFuncDefNode in the tree

    The assumption is that there's only one CFuncDefNode
    """
    found_node = None

    def visit_Node(self, node):
        if False:
            i = 10
            return i + 15
        if self.found_node:
            return
        else:
            self.visitchildren(node)

    def visit_CFuncDefNode(self, node):
        if False:
            return 10
        self.found_node = node

    def __call__(self, tree):
        if False:
            for i in range(10):
                print('nop')
        self.visit(tree)
        return self.found_node

def get_cfunc_from_tree(tree):
    if False:
        for i in range(10):
            print('nop')
    return _FindCFuncDefNode()(tree)

class _ArgumentInfo(object):
    """
    Everything related to defining an input/output argument for a ufunc

    type  - PyrexType
    type_constant  - str such as "NPY_INT8" representing numpy dtype constants
    """

    def __init__(self, type, type_constant):
        if False:
            for i in range(10):
                print('nop')
        self.type = type
        self.type_constant = type_constant

class UFuncConversion(object):

    def __init__(self, node):
        if False:
            while True:
                i = 10
        self.node = node
        self.global_scope = node.local_scope.global_scope()
        self.in_definitions = self.get_in_type_info()
        self.out_definitions = self.get_out_type_info()

    def get_in_type_info(self):
        if False:
            return 10
        definitions = []
        for (n, arg) in enumerate(self.node.args):
            type_const = _get_type_constant(self.node.pos, arg.type)
            definitions.append(_ArgumentInfo(arg.type, type_const))
        return definitions

    def get_out_type_info(self):
        if False:
            i = 10
            return i + 15
        if self.node.return_type.is_ctuple:
            components = self.node.return_type.components
        else:
            components = [self.node.return_type]
        definitions = []
        for (n, type) in enumerate(components):
            definitions.append(_ArgumentInfo(type, _get_type_constant(self.node.pos, type)))
        return definitions

    def generate_cy_utility_code(self):
        if False:
            for i in range(10):
                print('nop')
        arg_types = [a.type for a in self.in_definitions]
        out_types = [a.type for a in self.out_definitions]
        inline_func_decl = self.node.entry.type.declaration_code(self.node.entry.cname, pyrex=True)
        self.node.entry.used = True
        ufunc_cname = self.global_scope.next_id(self.node.entry.name + '_ufunc_def')
        will_be_called_without_gil = not (any((t.is_pyobject for t in arg_types)) or any((t.is_pyobject for t in out_types)))
        context = dict(func_cname=ufunc_cname, in_types=arg_types, out_types=out_types, inline_func_call=self.node.entry.cname, inline_func_declaration=inline_func_decl, nogil=self.node.entry.type.nogil, will_be_called_without_gil=will_be_called_without_gil)
        code = CythonUtilityCode.load('UFuncDefinition', 'UFuncs.pyx', context=context, outer_module_scope=self.global_scope)
        tree = code.get_tree(entries_only=True)
        return tree

    def use_generic_utility_code(self):
        if False:
            for i in range(10):
                print('nop')
        self.global_scope.use_utility_code(UtilityCode.load_cached('UFuncsInit', 'UFuncs_C.c'))
        self.global_scope.use_utility_code(UtilityCode.load_cached('NumpyImportUFunc', 'NumpyImportArray.c'))

def convert_to_ufunc(node):
    if False:
        print('Hello World!')
    if isinstance(node, Nodes.CFuncDefNode):
        if node.local_scope.parent_scope.is_c_class_scope:
            error(node.pos, 'Methods cannot currently be converted to a ufunc')
            return node
        converters = [UFuncConversion(node)]
        original_node = node
    elif isinstance(node, FusedNode.FusedCFuncDefNode) and isinstance(node.node, Nodes.CFuncDefNode):
        if node.node.local_scope.parent_scope.is_c_class_scope:
            error(node.pos, 'Methods cannot currently be converted to a ufunc')
            return node
        converters = [UFuncConversion(n) for n in node.nodes]
        original_node = node.node
    else:
        error(node.pos, 'Only C functions can be converted to a ufunc')
        return node
    if not converters:
        return
    del converters[0].global_scope.entries[original_node.entry.name]
    converters[0].use_generic_utility_code()
    return [node] + _generate_stats_from_converters(converters, original_node)

def generate_ufunc_initialization(converters, cfunc_nodes, original_node):
    if False:
        return 10
    global_scope = converters[0].global_scope
    ufunc_funcs_name = global_scope.next_id(Naming.pyrex_prefix + 'funcs')
    ufunc_types_name = global_scope.next_id(Naming.pyrex_prefix + 'types')
    ufunc_data_name = global_scope.next_id(Naming.pyrex_prefix + 'data')
    type_constants = []
    narg_in = None
    narg_out = None
    for c in converters:
        in_const = [d.type_constant for d in c.in_definitions]
        if narg_in is not None:
            assert narg_in == len(in_const)
        else:
            narg_in = len(in_const)
        type_constants.extend(in_const)
        out_const = [d.type_constant for d in c.out_definitions]
        if narg_out is not None:
            assert narg_out == len(out_const)
        else:
            narg_out = len(out_const)
        type_constants.extend(out_const)
    func_cnames = [cfnode.entry.cname for cfnode in cfunc_nodes]
    context = dict(ufunc_funcs_name=ufunc_funcs_name, func_cnames=func_cnames, ufunc_types_name=ufunc_types_name, type_constants=type_constants, ufunc_data_name=ufunc_data_name)
    global_scope.use_utility_code(TempitaUtilityCode.load('UFuncConsts', 'UFuncs_C.c', context=context))
    pos = original_node.pos
    func_name = original_node.entry.name
    docstr = original_node.doc
    args_to_func = '%s(), %s, %s(), %s, %s, %s, PyUFunc_None, "%s", %s, 0' % (ufunc_funcs_name, ufunc_data_name, ufunc_types_name, len(func_cnames), narg_in, narg_out, func_name, docstr.as_c_string_literal() if docstr else 'NULL')
    call_node = ExprNodes.PythonCapiCallNode(pos, function_name='PyUFunc_FromFuncAndData', func_type=PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dummy', PyrexTypes.c_void_ptr_type, None)]), args=[ExprNodes.ConstNode(pos, type=PyrexTypes.c_void_ptr_type, value=args_to_func)])
    lhs_entry = global_scope.declare_var(func_name, PyrexTypes.py_object_type, pos)
    assgn_node = Nodes.SingleAssignmentNode(pos, lhs=ExprNodes.NameNode(pos, name=func_name, type=PyrexTypes.py_object_type, entry=lhs_entry), rhs=call_node)
    return assgn_node

def _generate_stats_from_converters(converters, node):
    if False:
        while True:
            i = 10
    stats = []
    for converter in converters:
        tree = converter.generate_cy_utility_code()
        ufunc_node = get_cfunc_from_tree(tree)
        converter.global_scope.utility_code_list.extend(tree.scope.utility_code_list)
        stats.append(ufunc_node)
    stats.append(generate_ufunc_initialization(converters, stats, node))
    return stats