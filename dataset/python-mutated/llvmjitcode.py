"""
Use llvmlite to create executable functions from SymPy expressions

This module requires llvmlite (https://github.com/numba/llvmlite).
"""
import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
llvmlite = import_module('llvmlite')
if llvmlite:
    ll = import_module('llvmlite.ir').ir
    llvm = import_module('llvmlite.binding').binding
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
__doctest_requires__ = {'llvm_callable': ['llvmlite']}

class LLVMJitPrinter(Printer):
    """Convert expressions to LLVM IR"""

    def __init__(self, module, builder, fn, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.func_arg_map = kwargs.pop('func_arg_map', {})
        if not llvmlite:
            raise ImportError('llvmlite is required for LLVMJITPrinter')
        super().__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        if False:
            return 10
        self.tmp_var[name] = value

    def _print_Number(self, n):
        if False:
            print('Hello World!')
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        if False:
            print('Hello World!')
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        if False:
            while True:
                i = 10
        val = self.tmp_var.get(s)
        if not val:
            val = self.func_arg_map.get(s)
        if not val:
            raise LookupError('Symbol not found: %s' % s)
        return val

    def _print_Pow(self, expr):
        if False:
            i = 10
            return i + 15
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get('sqrt')
            if not fn:
                fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
                fn = ll.Function(self.module, fn_type, 'sqrt')
                self.ext_fn['sqrt'] = fn
            return self.builder.call(fn, [base0], 'sqrt')
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)
        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get('pow')
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ll.Function(self.module, fn_type, 'pow')
            self.ext_fn['pow'] = fn
        return self.builder.call(fn, [base0, exp0], 'pow')

    def _print_Mul(self, expr):
        if False:
            print('Hello World!')
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        if False:
            return 10
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

    def _print_Function(self, expr):
        if False:
            i = 10
            return i + 15
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            fn = ll.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def emptyPrinter(self, expr):
        if False:
            return 10
        raise TypeError('Unsupported type for LLVM JIT conversion: %s' % type(expr))

class LLVMJitCallbackPrinter(LLVMJitPrinter):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

    def _print_Indexed(self, expr):
        if False:
            return 10
        (array, idx) = self.func_arg_map[expr.base]
        offset = int(expr.indices[0].evalf())
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), offset)])
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value

    def _print_Symbol(self, s):
        if False:
            while True:
                i = 10
        val = self.tmp_var.get(s)
        if val:
            return val
        (array, idx) = self.func_arg_map.get(s, [None, 0])
        if not array:
            raise LookupError('Symbol not found: %s' % s)
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), idx)])
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value
exe_engines = []
link_names = set()
current_link_suffix = 0

class LLVMJitCode:

    def __init__(self, signature):
        if False:
            i = 10
            return i + 15
        self.signature = signature
        self.fp_type = ll.DoubleType()
        self.module = ll.Module('mod1')
        self.fn = None
        self.llvm_arg_types = []
        self.llvm_ret_type = self.fp_type
        self.param_dict = {}
        self.link_name = ''

    def _from_ctype(self, ctype):
        if False:
            i = 10
            return i + 15
        if ctype == ctypes.c_int:
            return ll.IntType(32)
        if ctype == ctypes.c_double:
            return self.fp_type
        if ctype == ctypes.POINTER(ctypes.c_double):
            return ll.PointerType(self.fp_type)
        if ctype == ctypes.c_void_p:
            return ll.PointerType(ll.IntType(32))
        if ctype == ctypes.py_object:
            return ll.PointerType(ll.IntType(32))
        print('Unhandled ctype = %s' % str(ctype))

    def _create_args(self, func_args):
        if False:
            for i in range(10):
                print('nop')
        'Create types for function arguments'
        self.llvm_ret_type = self._from_ctype(self.signature.ret_type)
        self.llvm_arg_types = [self._from_ctype(a) for a in self.signature.arg_ctypes]

    def _create_function_base(self):
        if False:
            i = 10
            return i + 15
        'Create function with name and type signature'
        global link_names, current_link_suffix
        default_link_name = 'jit_func'
        current_link_suffix += 1
        self.link_name = default_link_name + str(current_link_suffix)
        link_names.add(self.link_name)
        fn_type = ll.FunctionType(self.llvm_ret_type, self.llvm_arg_types)
        self.fn = ll.Function(self.module, fn_type, name=self.link_name)

    def _create_param_dict(self, func_args):
        if False:
            i = 10
            return i + 15
        'Mapping of symbolic values to function arguments'
        for (i, a) in enumerate(func_args):
            self.fn.args[i].name = str(a)
            self.param_dict[a] = self.fn.args[i]

    def _create_function(self, expr):
        if False:
            while True:
                i = 10
        'Create function body and return LLVM IR'
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)
        lj = LLVMJitPrinter(self.module, builder, self.fn, func_arg_map=self.param_dict)
        ret = self._convert_expr(lj, expr)
        lj.builder.ret(self._wrap_return(lj, ret))
        strmod = str(self.module)
        return strmod

    def _wrap_return(self, lj, vals):
        if False:
            while True:
                i = 10
        if self.signature.ret_type == ctypes.c_double:
            return vals[0]
        void_ptr = ll.PointerType(ll.IntType(32))
        wrap_type = ll.FunctionType(void_ptr, [self.fp_type])
        wrap_fn = ll.Function(lj.module, wrap_type, 'PyFloat_FromDouble')
        wrapped_vals = [lj.builder.call(wrap_fn, [v]) for v in vals]
        if len(vals) == 1:
            final_val = wrapped_vals[0]
        else:
            tuple_arg_types = [ll.IntType(32)]
            tuple_arg_types.extend([void_ptr] * len(vals))
            tuple_type = ll.FunctionType(void_ptr, tuple_arg_types)
            tuple_fn = ll.Function(lj.module, tuple_type, 'PyTuple_Pack')
            tuple_args = [ll.Constant(ll.IntType(32), len(wrapped_vals))]
            tuple_args.extend(wrapped_vals)
            final_val = lj.builder.call(tuple_fn, tuple_args)
        return final_val

    def _convert_expr(self, lj, expr):
        if False:
            return 10
        try:
            if len(expr) == 2:
                tmp_exprs = expr[0]
                final_exprs = expr[1]
                if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                    raise NotImplementedError('Return of multiple expressions not supported for this callback')
                for (name, e) in tmp_exprs:
                    val = lj._print(e)
                    lj._add_tmp_var(name, val)
        except TypeError:
            final_exprs = [expr]
        vals = [lj._print(e) for e in final_exprs]
        return vals

    def _compile_function(self, strmod):
        if False:
            for i in range(10):
                print('nop')
        global exe_engines
        llmod = llvm.parse_assembly(strmod)
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pass_manager = llvm.create_module_pass_manager()
        pmb.populate(pass_manager)
        pass_manager.run(llmod)
        target_machine = llvm.Target.from_default_triple().create_target_machine()
        exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
        exe_eng.finalize_object()
        exe_engines.append(exe_eng)
        if False:
            print('Assembly')
            print(target_machine.emit_assembly(llmod))
        fptr = exe_eng.get_function_address(self.link_name)
        return fptr

class LLVMJitCodeCallback(LLVMJitCode):

    def __init__(self, signature):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(signature)

    def _create_param_dict(self, func_args):
        if False:
            while True:
                i = 10
        for (i, a) in enumerate(func_args):
            if isinstance(a, IndexedBase):
                self.param_dict[a] = (self.fn.args[i], i)
                self.fn.args[i].name = str(a)
            else:
                self.param_dict[a] = (self.fn.args[self.signature.input_arg], i)

    def _create_function(self, expr):
        if False:
            i = 10
            return i + 15
        'Create function body and return LLVM IR'
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)
        lj = LLVMJitCallbackPrinter(self.module, builder, self.fn, func_arg_map=self.param_dict)
        ret = self._convert_expr(lj, expr)
        if self.signature.ret_arg:
            output_fp_ptr = builder.bitcast(self.fn.args[self.signature.ret_arg], ll.PointerType(self.fp_type))
            for (i, val) in enumerate(ret):
                index = ll.Constant(ll.IntType(32), i)
                output_array_ptr = builder.gep(output_fp_ptr, [index])
                builder.store(val, output_array_ptr)
            builder.ret(ll.Constant(ll.IntType(32), 0))
        else:
            lj.builder.ret(self._wrap_return(lj, ret))
        strmod = str(self.module)
        return strmod

class CodeSignature:

    def __init__(self, ret_type):
        if False:
            return 10
        self.ret_type = ret_type
        self.arg_ctypes = []
        self.input_arg = 0
        self.ret_arg = None

def _llvm_jit_code(args, expr, signature, callback_type):
    if False:
        i = 10
        return i + 15
    'Create a native code function from a SymPy expression'
    if callback_type is None:
        jit = LLVMJitCode(signature)
    else:
        jit = LLVMJitCodeCallback(signature)
    jit._create_args(args)
    jit._create_function_base()
    jit._create_param_dict(args)
    strmod = jit._create_function(expr)
    if False:
        print('LLVM IR')
        print(strmod)
    fptr = jit._compile_function(strmod)
    return fptr

@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    if False:
        print('Hello World!')
    "Compile function from a SymPy expression\n\n    Expressions are evaluated using double precision arithmetic.\n    Some single argument math functions (exp, sin, cos, etc.) are supported\n    in expressions.\n\n    Parameters\n    ==========\n\n    args : List of Symbol\n        Arguments to the generated function.  Usually the free symbols in\n        the expression.  Currently each one is assumed to convert to\n        a double precision scalar.\n    expr : Expr, or (Replacements, Expr) as returned from 'cse'\n        Expression to compile.\n    callback_type : string\n        Create function with signature appropriate to use as a callback.\n        Currently supported:\n           'scipy.integrate'\n           'scipy.integrate.test'\n           'cubature'\n\n    Returns\n    =======\n\n    Compiled function that can evaluate the expression.\n\n    Examples\n    ========\n\n    >>> import sympy.printing.llvmjitcode as jit\n    >>> from sympy.abc import a\n    >>> e = a*a + a + 1\n    >>> e1 = jit.llvm_callable([a], e)\n    >>> e.subs(a, 1.1)   # Evaluate via substitution\n    3.31000000000000\n    >>> e1(1.1)  # Evaluate using JIT-compiled code\n    3.3100000000000005\n\n\n    Callbacks for integration functions can be JIT compiled.\n    >>> import sympy.printing.llvmjitcode as jit\n    >>> from sympy.abc import a\n    >>> from sympy import integrate\n    >>> from scipy.integrate import quad\n    >>> e = a*a\n    >>> e1 = jit.llvm_callable([a], e, callback_type='scipy.integrate')\n    >>> integrate(e, (a, 0.0, 2.0))\n    2.66666666666667\n    >>> quad(e1, 0.0, 2.0)[0]\n    2.66666666666667\n\n    The 'cubature' callback is for the Python wrapper around the\n    cubature package ( https://github.com/saullocastro/cubature )\n    and ( http://ab-initio.mit.edu/wiki/index.php/Cubature )\n\n    There are two signatures for the SciPy integration callbacks.\n    The first ('scipy.integrate') is the function to be passed to the\n    integration routine, and will pass the signature checks.\n    The second ('scipy.integrate.test') is only useful for directly calling\n    the function using ctypes variables. It will not pass the signature checks\n    for scipy.integrate.\n\n    The return value from the cse module can also be compiled.  This\n    can improve the performance of the compiled function.  If multiple\n    expressions are given to cse, the compiled function returns a tuple.\n    The 'cubature' callback handles multiple expressions (set `fdim`\n    to match in the integration call.)\n    >>> import sympy.printing.llvmjitcode as jit\n    >>> from sympy import cse\n    >>> from sympy.abc import x,y\n    >>> e1 = x*x + y*y\n    >>> e2 = 4*(x*x + y*y) + 8.0\n    >>> after_cse = cse([e1,e2])\n    >>> after_cse\n    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])\n    >>> j1 = jit.llvm_callable([x,y], after_cse)\n    >>> j1(1.0, 2.0)\n    (5.0, 28.0)\n    "
    if not llvmlite:
        raise ImportError('llvmlite is required for llvmjitcode')
    signature = CodeSignature(ctypes.py_object)
    arg_ctypes = []
    if callback_type is None:
        for _ in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    elif callback_type in ('scipy.integrate', 'scipy.integrate.test'):
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        signature.ret_type = ctypes.c_int
        signature.input_arg = 1
        signature.ret_arg = 4
    else:
        raise ValueError('Unknown callback type: %s' % callback_type)
    signature.arg_ctypes = arg_ctypes
    fptr = _llvm_jit_code(args, expr, signature, callback_type)
    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal
    if signature.ret_type == ctypes.py_object:
        FUNCTYPE = ctypes.PYFUNCTYPE
    else:
        FUNCTYPE = ctypes.CFUNCTYPE
    cfunc = FUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    return cfunc