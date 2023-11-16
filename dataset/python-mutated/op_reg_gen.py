"""op_reg_gen: Generate op registration code from composite op code."""
import gast as ast
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.framework import op_def_registry
from tensorflow.python.util import tf_inspect
_COMPOSITE_ARG_LIST = ['op_name', 'inputs', 'attrs', 'derived_attrs', 'outputs']

class OpRegGenImpl(transformer.CodeGenerator):
    """Visit the AST and generate C++ op registration functions."""

    def __init__(self, ctx):
        if False:
            i = 10
            return i + 15
        super(OpRegGenImpl, self).__init__(ctx)
        self.ctx = ctx

    def visit_Name(self, node):
        if False:
            for i in range(10):
                print('nop')
        return node.id

    def visit_Constant(self, node):
        if False:
            i = 10
            return i + 15
        return node.value

    def visit_keyword(self, node):
        if False:
            i = 10
            return i + 15
        return (node.arg, self.visit(node.value))

    def visit_List(self, node):
        if False:
            i = 10
            return i + 15
        return [self.visit(cst) for cst in node.elts]

    def visit_arguments(self, node):
        if False:
            for i in range(10):
                print('nop')
        return [self.visit(arg) for arg in node.args]

    def visit_FunctionDef(self, node):
        if False:
            for i in range(10):
                print('nop')
        compose_dec = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Attribute) and dec.func.attr == 'Composite':
                    compose_dec.append(dec)
                if isinstance(dec.func, ast.Name) and dec.func.id == 'Composite':
                    compose_dec.append(dec)
        if not compose_dec:
            return
        elif len(compose_dec) > 1:
            raise KeyError('More than one TF ops decomposes for.')
        all_dec_args = {}
        for (arg_name, arg_value) in zip(_COMPOSITE_ARG_LIST, compose_dec[0].args):
            all_dec_args[arg_name] = self.visit(arg_value)
        kw_dec_args = dict([self.visit(kw) for kw in compose_dec[0].keywords])
        if all_dec_args.keys() & kw_dec_args.keys():
            raise KeyError('More arguments than expected.')
        all_dec_args.update(kw_dec_args)
        op_name = all_dec_args['op_name']
        op_def = op_def_registry.get(op_name)
        if op_def:
            if len(all_dec_args) > 1:
                raise ValueError('Op has been registered: ' + op_name)
            else:
                return
        inputs = all_dec_args.get('inputs', [])
        attrs = all_dec_args.get('attrs', [])
        expected_args = [arg.split(':')[0] for arg in inputs + attrs]
        all_func_args = self.visit(node.args)
        if len(expected_args) != len(all_func_args):
            raise KeyError('Composition arguments for {} do not match the registration. {} vs {}'.format(op_name, expected_args, all_func_args))
        cxx_reg_code = ['\nREGISTER_OP("{}")'.format(op_name)]
        for input_ in inputs:
            cxx_reg_code.append('.Input("{}")'.format(input_))
        for attr in attrs:
            py_str = attr.replace('"', "'")
            cxx_reg_code.append('.Attr("{}")'.format(py_str))
        for attr in all_dec_args.get('derived_attrs', []):
            py_str = attr.replace('"', "'")
            cxx_reg_code.append('.Attr("{}")'.format(py_str))
        for output_ in all_dec_args.get('outputs', []):
            cxx_reg_code.append('.Output("{}")'.format(output_))
        cxx_reg_code[-1] += ';\n'
        self.emit('\n    '.join(cxx_reg_code))

class OpRegGen(transpiler.GenericTranspiler):
    """Transforms Python objects into TFR MLIR source code."""

    def transform_ast(self, node, ctx):
        if False:
            i = 10
            return i + 15
        gen = OpRegGenImpl(ctx)
        gen.visit(node)
        return gen.code_buffer

def op_reg_gen(func):
    if False:
        i = 10
        return i + 15
    'Parse a function and emit the TFR functions.'
    (op_reg_code, _) = OpRegGen().transform(func, None)
    return op_reg_code

def gen_register_op(source, method_prefix=None):
    if False:
        for i in range(10):
            print('nop')
    'Parse a python code and emit the TFR functions from a target class.'
    mlir_funcs = [op_reg_gen(func) for (name, func) in tf_inspect.getmembers(source, tf_inspect.isfunction) if not method_prefix or name.startswith(method_prefix)]
    headers = '\n#include "tensorflow/core/framework/op.h"\n\nnamespace tensorflow {\n  '
    code = '\n'.join(mlir_funcs)
    return headers + code + '}  // namespace tensorflow\n'