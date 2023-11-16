from __future__ import annotations
import ast
import importlib.util
import itertools
import marshal
import os
import sys
from ast import AST, ClassDef
from builtins import compile as builtin_compile
from contextlib import contextmanager
from typing import Dict, Union
from . import consts, future, misc, pyassem, symbols
from .consts import CO_ASYNC_GENERATOR, CO_COROUTINE, CO_GENERATOR, CO_NESTED, CO_VARARGS, CO_VARKEYWORDS, PyCF_MASK_OBSOLETE, PyCF_ONLY_AST, PyCF_SOURCE_IS_UTF8, SC_CELL, SC_FREE, SC_GLOBAL_EXPLICIT, SC_GLOBAL_IMPLICIT, SC_LOCAL
from .optimizer import AstOptimizer
from .pyassem import Block, PyFlowGraph
from .symbols import Scope, SymbolVisitor
from .unparse import to_expr
from .visitor import ASTVisitor, walk
TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Generator, List, Optional, Sequence, Type, Union
try:
    from cinder import _set_qualname
except ImportError:

    def _set_qualname(code, qualname):
        if False:
            i = 10
            return i + 15
        pass
callfunc_opcode_info = {(0, 0): 'CALL_FUNCTION', (1, 0): 'CALL_FUNCTION_VAR', (0, 1): 'CALL_FUNCTION_KW', (1, 1): 'CALL_FUNCTION_VAR_KW'}
INT_MAX = 2 ** 31 - 1
WHILE_LOOP = 1
FOR_LOOP = 2
TRY_EXCEPT = 3
FINALLY_TRY = 4
FINALLY_END = 5
WITH = 6
ASYNC_WITH = 7
HANDLER_CLEANUP = 8
POP_VALUE = 9
EXCEPTION_HANDLER = 10
ASYNC_COMPREHENSION_GENERATOR = 11
_ZERO = 0 .to_bytes(4, 'little')
_DEFAULT_MODNAME = sys.intern('<module>')
FuncOrLambda = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda]
CompNode = Union[ast.SetComp, ast.DictComp, ast.ListComp]
STACK_USE_GUIDELINE = 30

def make_header(mtime, size):
    if False:
        while True:
            i = 10
    return _ZERO + mtime.to_bytes(4, 'little') + size.to_bytes(4, 'little')

def compileFile(filename, display=0, compiler=None, modname=_DEFAULT_MODNAME):
    if False:
        i = 10
        return i + 15
    fileinfo = os.stat(filename)
    with open(filename, 'U') as f:
        buf = f.read()
    code = compile(buf, filename, 'exec', compiler=compiler, modname=modname)
    with open(filename + 'c', 'wb') as f:
        hdr = make_header(int(fileinfo.st_mtime), fileinfo.st_size)
        f.write(importlib.util.MAGIC_NUMBER)
        f.write(hdr)
        marshal.dump(code, f)

def compile(source, filename, mode, flags=0, dont_inherit=None, optimize=-1, compiler=None, modname=_DEFAULT_MODNAME):
    if False:
        print('Hello World!')
    'Replacement for builtin compile() function\n\n    Does not yet support ast.PyCF_ALLOW_TOP_LEVEL_AWAIT flag.\n    '
    if dont_inherit is not None:
        raise RuntimeError('not implemented yet')
    result = make_compiler(source, filename, mode, flags, optimize, compiler, modname)
    if flags & PyCF_ONLY_AST:
        return result
    return result.getCode()

def parse(source, filename, mode, flags):
    if False:
        print('Hello World!')
    return builtin_compile(source, filename, mode, flags | PyCF_ONLY_AST)

def make_compiler(source, filename, mode, flags=0, optimize=-1, generator=None, modname=_DEFAULT_MODNAME, ast_optimizer_enabled=True):
    if False:
        i = 10
        return i + 15
    if mode not in ('single', 'exec', 'eval'):
        raise ValueError("compile() mode must be 'exec', 'eval' or 'single'")
    if generator is None:
        generator = get_default_generator()
    if flags & ~(consts.PyCF_MASK | PyCF_MASK_OBSOLETE | consts.PyCF_COMPILE_MASK):
        raise ValueError('compile(): unrecognised flags', hex(flags))
    flags |= PyCF_SOURCE_IS_UTF8
    if isinstance(source, ast.AST):
        tree = source
    else:
        tree = parse(source, filename, mode, flags & consts.PyCF_MASK)
    if flags & PyCF_ONLY_AST:
        return tree
    optimize = sys.flags.optimize if optimize == -1 else optimize
    return generator.make_code_gen(modname, tree, filename, flags, optimize, ast_optimizer_enabled=ast_optimizer_enabled)

def is_const(node):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(node, ast.Constant)

def all_items_const(seq, begin, end):
    if False:
        while True:
            i = 10
    for item in seq[begin:end]:
        if not is_const(item):
            return False
    return True

def find_futures(flags: int, node: ast.Module) -> int:
    if False:
        for i in range(10):
            print('nop')
    future_flags = flags & consts.PyCF_MASK
    for feature in future.find_futures(node):
        if feature == 'barry_as_FLUFL':
            future_flags |= consts.CO_FUTURE_BARRY_AS_BDFL
        elif feature == 'annotations':
            future_flags |= consts.CO_FUTURE_ANNOTATIONS
    return future_flags
CONV_STR = ord('s')
CONV_REPR = ord('r')
CONV_ASCII = ord('a')

class PatternContext:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.stores: list[str] = []
        self.allow_irrefutable: bool = False
        self.fail_pop: list[Block] = []
        self.on_top: int = 0

    def clone(self) -> PatternContext:
        if False:
            i = 10
            return i + 15
        pc = PatternContext()
        pc.stores = list(self.stores)
        pc.allow_irrefutable = self.allow_irrefutable
        pc.fail_pop = list(self.fail_pop)
        pc.on_top = self.on_top
        return pc

class CodeGenerator(ASTVisitor):
    """Defines basic code generator for Python bytecode

    This class is an abstract base class.  Concrete subclasses must
    define an __init__() that defines self.graph and then calls the
    __init__() defined in this class.
    """
    optimized = 0
    __initialized = None
    class_name = None
    future_flags = 0
    flow_graph = pyassem.PyFlowGraph
    _SymbolVisitor = symbols.SymbolVisitor

    def __init__(self, parent: Optional[CodeGenerator], node: AST, symbols: SymbolVisitor, graph: PyFlowGraph, flags=0, optimization_lvl=0, future_flags=None):
        if False:
            return 10
        super().__init__()
        if parent is not None:
            assert future_flags is None, 'Child codegen should inherit future flags'
            future_flags = parent.future_flags
        self.future_flags = future_flags or 0
        graph.setFlag(self.future_flags)
        self.module_gen = self if parent is None else parent.module_gen
        self.tree = node
        self.symbols = symbols
        self.graph = graph
        self.scopes = symbols.scopes
        self.setups = misc.Stack()
        self.last_lineno = None
        self._setupGraphDelegation()
        self.interactive = False
        self.scope = self.scopes[node]
        self.flags = flags
        self.optimization_lvl = optimization_lvl
        self.strip_docstrings = optimization_lvl == 2
        self.__with_count = 0
        self.did_setup_annotations = False
        self._qual_name = None
        self.parent_code_gen = parent

    def _setupGraphDelegation(self):
        if False:
            for i in range(10):
                print('nop')
        self.emit = self.graph.emit
        self.emitWithBlock = self.graph.emitWithBlock
        self.emit_noline = self.graph.emit_noline
        self.newBlock = self.graph.newBlock
        self.nextBlock = self.graph.nextBlock

    def getCode(self):
        if False:
            i = 10
            return i + 15
        'Return a code object'
        return self.graph.getCode()

    def set_qual_name(self, qualname):
        if False:
            return 10
        pass

    @contextmanager
    def noEmit(self):
        if False:
            while True:
                i = 10
        self.graph.do_not_emit_bytecode += 1
        try:
            yield
        finally:
            self.graph.do_not_emit_bytecode -= 1

    @contextmanager
    def noOp(self):
        if False:
            print('Hello World!')
        yield

    def maybeEmit(self, test):
        if False:
            while True:
                i = 10
        return self.noOp() if test else self.noEmit()

    def mangle(self, name):
        if False:
            i = 10
            return i + 15
        if self.class_name is not None:
            return misc.mangle(name, self.class_name)
        else:
            return name

    def get_module(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('should be implemented by subclasses')

    def storeName(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._nameOp('STORE', name)

    def loadName(self, name):
        if False:
            print('Hello World!')
        self._nameOp('LOAD', name)

    def delName(self, name):
        if False:
            print('Hello World!')
        self._nameOp('DELETE', name)

    def _nameOp(self, prefix, name):
        if False:
            for i in range(10):
                print('nop')
        if prefix == 'LOAD' and name == 'super' and isinstance(self.scope, symbols.FunctionScope):
            scope = self.scope.check_name(name)
            if scope in (SC_GLOBAL_EXPLICIT, SC_GLOBAL_IMPLICIT):
                self.scope.suppress_jit = True
        name = self.mangle(name)
        scope = self.scope.check_name(name)
        if scope == SC_LOCAL:
            if not self.optimized:
                self.emit(prefix + '_NAME', name)
            else:
                self.emit(prefix + '_FAST', name)
        elif scope == SC_GLOBAL_EXPLICIT:
            self.emit(prefix + '_GLOBAL', name)
        elif scope == SC_GLOBAL_IMPLICIT:
            if not self.optimized:
                self.emit(prefix + '_NAME', name)
            else:
                self.emit(prefix + '_GLOBAL', name)
        elif scope == SC_FREE or scope == SC_CELL:
            if isinstance(self.scope, symbols.ClassScope):
                if prefix == 'STORE' and name not in self.scope.nonlocals:
                    self.emit(prefix + '_NAME', name)
                    return
            if isinstance(self.scope, symbols.ClassScope) and prefix == 'LOAD':
                self.emit(prefix + '_CLASSDEREF', name)
            else:
                self.emit(prefix + '_DEREF', name)
        else:
            raise RuntimeError('unsupported scope for var %s: %d' % (name, scope))

    def _implicitNameOp(self, prefix, name):
        if False:
            return 10
        "Emit name ops for names generated implicitly by for loops\n\n        The interpreter generates names that start with a period or\n        dollar sign.  The symbol table ignores these names because\n        they aren't present in the program text.\n        "
        if self.optimized:
            self.emit(prefix + '_FAST', name)
        else:
            self.emit(prefix + '_NAME', name)

    def set_lineno(self, node: AST):
        if False:
            print('Hello World!')
        self.graph.set_lineno(node.lineno)

    def set_no_lineno(self):
        if False:
            return 10
        'Mark following instructions as synthetic (no source line number).'
        self.graph.set_lineno(-1)

    @contextmanager
    def temp_lineno(self, lineno: int) -> Generator[None, None, None]:
        if False:
            return 10
        old_lineno = self.graph.lineno
        self.graph.set_lineno(lineno)
        try:
            yield
        finally:
            self.graph.set_lineno(old_lineno)

    def skip_docstring(self, body):
        if False:
            while True:
                i = 10
        'Given list of statements, representing body of a function, class,\n        or module, skip docstring, if any.\n        '
        if body and isinstance(body, list) and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            return body[1:]
        return body

    def visitInteractive(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.interactive = True
        self.visitStatements(node.body)
        self.emit('LOAD_CONST', None)
        self.emit('RETURN_VALUE')

    def visitModule(self, node: ast.Module) -> None:
        if False:
            for i in range(10):
                print('nop')
        if node.body:
            self.set_lineno(node.body[0])
        if self.findAnn(node.body):
            self.emit('SETUP_ANNOTATIONS')
            self.did_setup_annotations = True
        doc = self.get_docstring(node)
        if doc is not None:
            self.emit('LOAD_CONST', doc)
            self.storeName('__doc__')
        self.startModule()
        self.visitStatements(self.skip_docstring(node.body))
        if not self.graph.first_inst_lineno:
            self.graph.first_inst_lineno = 1
        self.emit_module_return(node)

    def startModule(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def emit_module_return(self, node: ast.Module) -> None:
        if False:
            while True:
                i = 10
        self.set_no_lineno()
        self.emit('LOAD_CONST', None)
        self.emit('RETURN_VALUE')

    def visitExpression(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.body)
        self.emit('RETURN_VALUE')

    def visitFunctionDef(self, node):
        if False:
            return 10
        self.visitFunctionOrLambda(node)
    visitAsyncFunctionDef = visitFunctionDef

    def visitJoinedStr(self, node):
        if False:
            i = 10
            return i + 15
        if len(node.values) > STACK_USE_GUIDELINE:
            self.emit('LOAD_CONST', '')
            self.emit('LOAD_METHOD', 'join')
            self.emit('BUILD_LIST')
            for value in node.values:
                self.visit(value)
                self.emit('LIST_APPEND', 1)
            self.emit('CALL_METHOD', 1)
        else:
            for value in node.values:
                self.visit(value)
            if len(node.values) != 1:
                self.emit('BUILD_STRING', len(node.values))

    def visitFormattedValue(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.value)
        if node.conversion == CONV_STR:
            oparg = pyassem.FVC_STR
        elif node.conversion == CONV_REPR:
            oparg = pyassem.FVC_REPR
        elif node.conversion == CONV_ASCII:
            oparg = pyassem.FVC_ASCII
        else:
            assert node.conversion == -1, str(node.conversion)
            oparg = pyassem.FVC_NONE
        if node.format_spec:
            self.visit(node.format_spec)
            oparg |= pyassem.FVS_HAVE_SPEC
        self.emit('FORMAT_VALUE', oparg)

    def visitLambda(self, node):
        if False:
            i = 10
            return i + 15
        self.visitFunctionOrLambda(node)

    def processBody(self, node, body, gen):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(body, list):
            for stmt in body:
                gen.visit(stmt)
        else:
            gen.visit(body)

    def _visitAnnotation(self, node):
        if False:
            while True:
                i = 10
        if self.future_flags & consts.CO_FUTURE_ANNOTATIONS:
            self.emit('LOAD_CONST', to_expr(node))
        else:
            self.visit(node)

    def generate_function(self, node: FuncOrLambda, name: str, first_lineno: int) -> CodeGenerator:
        if False:
            while True:
                i = 10
        gen = self.make_func_codegen(node, node.args, name, first_lineno)
        body = self.skip_docstring(node.body)
        self.processBody(node, body, gen)
        gen.finishFunction()
        return gen

    def build_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, gen: CodeGenerator) -> None:
        if False:
            return 10
        flags = 0
        if node.args.defaults:
            for default in node.args.defaults:
                self.visitDefault(default)
                flags |= 1
            self.emit('BUILD_TUPLE', len(node.args.defaults))
        kwdefaults = []
        for (kwonly, default) in zip(node.args.kwonlyargs, node.args.kw_defaults):
            if default is not None:
                kwdefaults.append(self.mangle(kwonly.arg))
                self.visitDefault(default)
        if kwdefaults:
            self.emit('LOAD_CONST', tuple(kwdefaults))
            self.emit('BUILD_CONST_KEY_MAP', len(kwdefaults))
            flags |= 2
        if self.build_annotations(node):
            flags |= 4
        self._makeClosure(gen, flags)

    def build_annotations(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> bool:
        if False:
            return 10
        annotation_count = self.annotate_args(node.args)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            returns = node.returns
            if returns:
                self.emit('LOAD_CONST', 'return')
                self._visitAnnotation(returns)
                annotation_count += 2
        if annotation_count > 0:
            self.emit('BUILD_TUPLE', annotation_count)
        return annotation_count > 0

    def visitFunctionOrLambda(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        if False:
            print('Hello World!')
        if isinstance(node, ast.Lambda):
            name = sys.intern('<lambda>')
            (ndecorators, first_lineno) = (0, node.lineno)
        else:
            name = node.name
            if node.decorator_list:
                for decorator in node.decorator_list:
                    self.visit(decorator)
                ndecorators = len(node.decorator_list)
                first_lineno = node.decorator_list[0].lineno
            else:
                ndecorators = 0
                first_lineno = node.lineno
        gen = self.generate_function(node, name, first_lineno)
        self.build_function(node, gen)
        for _ in range(ndecorators):
            self.emit('CALL_FUNCTION', 1)
        if not isinstance(node, ast.Lambda):
            self.storeName(gen.graph.name)

    def visitDefault(self, node: ast.expr) -> None:
        if False:
            while True:
                i = 10
        self.visit(node)

    def annotate_args(self, args: ast.arguments) -> int:
        if False:
            return 10
        annotation_count = 0
        for arg in args.args:
            annotation_count += self.annotate_arg(arg)
        for arg in args.posonlyargs:
            annotation_count += self.annotate_arg(arg)
        if (arg := args.vararg):
            annotation_count += self.annotate_arg(arg)
        for arg in args.kwonlyargs:
            annotation_count += self.annotate_arg(arg)
        if (arg := args.kwarg):
            annotation_count += self.annotate_arg(arg)
        return annotation_count

    def annotate_arg(self, arg: ast.arg) -> int:
        if False:
            i = 10
            return i + 15
        ann = arg.annotation
        if ann:
            self.emit('LOAD_CONST', self.mangle(arg.arg))
            self._visitAnnotation(ann)
            return 2
        return 0

    def visitClassDef(self, node: ast.ClassDef) -> None:
        if False:
            print('Hello World!')
        first_lineno = None
        immutability_flag = self.find_immutability_flag(node)
        for decorator in node.decorator_list:
            if first_lineno is None:
                first_lineno = decorator.lineno
            self.visit_decorator(decorator, node)
        first_lineno = node.lineno if first_lineno is None else first_lineno
        gen = self.make_class_codegen(node, first_lineno)
        gen.emit('LOAD_NAME', '__name__')
        gen.storeName('__module__')
        gen.emit('LOAD_CONST', gen.get_qual_prefix(gen) + gen.name)
        gen.storeName('__qualname__')
        if gen.findAnn(node.body):
            gen.did_setup_annotations = True
            gen.emit('SETUP_ANNOTATIONS')
        doc = gen.get_docstring(node)
        if doc is not None:
            gen.set_lineno(node.body[0])
            gen.emit('LOAD_CONST', doc)
            gen.storeName('__doc__')
        self.walkClassBody(node, gen)
        gen.set_no_lineno()
        if '__class__' in gen.scope.cells:
            gen.emit('LOAD_CLOSURE', '__class__')
            gen.emit('DUP_TOP')
            gen.emit('STORE_NAME', '__classcell__')
        else:
            gen.emit('LOAD_CONST', None)
        gen.emit('RETURN_VALUE')
        self.emit_build_class(node, gen)
        for d in reversed(node.decorator_list):
            self.emit_decorator_call(d, node)
        self.register_immutability(node, immutability_flag)
        self.post_process_and_store_name(node)

    def emit_build_class(self, node: ast.ClassDef, class_body: CodeGenerator):
        if False:
            i = 10
            return i + 15
        self.emit('LOAD_BUILD_CLASS')
        self._makeClosure(class_body, 0)
        self.emit('LOAD_CONST', node.name)
        self._call_helper(2, None, node.bases, node.keywords)

    def find_immutability_flag(self, node: ClassDef) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def visit_decorator(self, decorator: AST, class_def: ClassDef) -> None:
        if False:
            return 10
        self.visit(decorator)

    def emit_decorator_call(self, decorator: AST, class_def: ClassDef) -> None:
        if False:
            return 10
        self.emit('CALL_FUNCTION', 1)

    def register_immutability(self, node: ClassDef, flag: bool) -> None:
        if False:
            print('Hello World!')
        '\n        Note: make sure this do not have side effect on the\n        stack, assumes class is on the stack\n        '

    def post_process_and_store_name(self, node: ClassDef) -> None:
        if False:
            print('Hello World!')
        self.storeName(node.name)

    def walkClassBody(self, node: ClassDef, gen: 'CodeGenerator'):
        if False:
            i = 10
            return i + 15
        walk(self.skip_docstring(node.body), gen)

    def visitIf(self, node):
        if False:
            return 10
        end = self.newBlock('if_end')
        orelse = None
        if node.orelse:
            orelse = self.newBlock('if_else')
        self.compileJumpIf(node.test, orelse or end, False)
        self.visitStatements(node.body)
        if node.orelse:
            self.emit_noline('JUMP_FORWARD', end)
            self.nextBlock(orelse)
            self.visitStatements(node.orelse)
        self.nextBlock(end)

    def visitWhile(self, node):
        if False:
            return 10
        loop = self.newBlock('while_loop')
        body = self.newBlock('while_body')
        else_ = self.newBlock('while_else')
        after = self.newBlock('while_after')
        self.push_loop(WHILE_LOOP, loop, after)
        self.nextBlock(loop)
        self.compileJumpIf(node.test, else_, False)
        self.nextBlock(body)
        self.visitStatements(node.body)
        self.set_lineno(node)
        self.compileJumpIf(node.test, body, True)
        self.pop_loop()
        self.nextBlock(else_)
        if node.orelse:
            self.visitStatements(node.orelse)
        self.nextBlock(after)

    def push_loop(self, kind, start, end):
        if False:
            i = 10
            return i + 15
        self.setups.push(Entry(kind, start, end, None))

    def pop_loop(self):
        if False:
            print('Hello World!')
        self.setups.pop()

    def visitFor(self, node):
        if False:
            while True:
                i = 10
        start = self.newBlock('for_start')
        body = self.newBlock('for_body')
        cleanup = self.newBlock('for_cleanup')
        end = self.newBlock('for_end')
        self.push_loop(FOR_LOOP, start, end)
        self.visit(node.iter)
        self.emit('GET_ITER')
        self.nextBlock(start)
        self.emit('FOR_ITER', cleanup)
        self.nextBlock(body)
        self.visit(node.target)
        self.visitStatements(node.body)
        self.set_no_lineno()
        self.emit('JUMP_ABSOLUTE', start)
        self.nextBlock(cleanup)
        self.pop_loop()
        if node.orelse:
            self.visitStatements(node.orelse)
        self.nextBlock(end)

    def visitAsyncFor(self, node):
        if False:
            return 10
        start = self.newBlock('async_for_try')
        except_ = self.newBlock('except')
        end = self.newBlock('end')
        self.visit(node.iter)
        self.emit('GET_AITER')
        self.nextBlock(start)
        self.setups.push(Entry(FOR_LOOP, start, end, None))
        self.emit('SETUP_FINALLY', except_)
        self.emit('GET_ANEXT')
        self.emit('LOAD_CONST', None)
        self.emit('YIELD_FROM')
        self.emit('POP_BLOCK')
        self.visit(node.target)
        self.visitStatements(node.body)
        self.set_no_lineno()
        self.emit('JUMP_ABSOLUTE', start)
        self.setups.pop()
        self.nextBlock(except_)
        self.set_lineno(node.iter)
        self.emit('END_ASYNC_FOR')
        if node.orelse:
            self.visitStatements(node.orelse)
        self.nextBlock(end)

    def visitBreak(self, node):
        if False:
            print('Hello World!')
        self.emit('NOP')
        loop = self.unwind_setup_entries(preserve_tos=False, stop_on_loop=True)
        if loop is None:
            raise SyntaxError("'break' outside loop", self.syntax_error_position(node))
        self.unwind_setup_entry(loop, preserve_tos=False)
        self.emit('JUMP_ABSOLUTE', loop.exit)
        self.nextBlock()

    def visitContinue(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.emit('NOP')
        loop = self.unwind_setup_entries(preserve_tos=False, stop_on_loop=True)
        if loop is None:
            raise SyntaxError("'continue' not properly in loop", self.syntax_error_position(node))
        self.emit('JUMP_ABSOLUTE', loop.block)
        self.nextBlock()

    def syntax_error_position(self, node):
        if False:
            for i in range(10):
                print('nop')
        import linecache
        source_line = linecache.getline(self.graph.filename, node.lineno)
        return (self.graph.filename, node.lineno, node.col_offset, source_line or None)

    def syntax_error(self, msg, node):
        if False:
            while True:
                i = 10
        import linecache
        source_line = linecache.getline(self.graph.filename, node.lineno)
        return SyntaxError(msg, (self.graph.filename, node.lineno, node.col_offset, source_line or None))

    def compileJumpIfPop(self, test, label, is_if_true):
        if False:
            for i in range(10):
                print('nop')
        self.visit(test)
        self.emit('JUMP_IF_TRUE_OR_POP' if is_if_true else 'JUMP_IF_FALSE_OR_POP', label)

    def visitTest(self, node, is_if_true: bool):
        if False:
            for i in range(10):
                print('nop')
        end = self.newBlock()
        for child in node.values[:-1]:
            self.compileJumpIfPop(child, end, is_if_true)
            self.nextBlock()
        self.visit(node.values[-1])
        self.nextBlock(end)

    def visitBoolOp(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visitTest(node, type(node.op) == ast.Or)
    _cmp_opcode: Dict[Type, str] = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>='}

    def compileJumpIf(self, test: ast.expr, next: Block, is_if_true: bool) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(test, ast.UnaryOp):
            if isinstance(test.op, ast.Not):
                self.compileJumpIf(test.operand, next, not is_if_true)
                return
        elif isinstance(test, ast.BoolOp):
            is_or = isinstance(test.op, ast.Or)
            skip_jump = next
            if is_if_true != is_or:
                skip_jump = self.newBlock()
            for node in test.values[:-1]:
                self.compileJumpIf(node, skip_jump, is_or)
            self.compileJumpIf(test.values[-1], next, is_if_true)
            if skip_jump is not next:
                self.nextBlock(skip_jump)
            return
        elif isinstance(test, ast.IfExp):
            end = self.newBlock('end')
            orelse = self.newBlock('orelse')
            self.compileJumpIf(test.test, orelse, False)
            self.compileJumpIf(test.body, next, is_if_true)
            self.emit_noline('JUMP_FORWARD', end)
            self.nextBlock(orelse)
            self.compileJumpIf(test.orelse, next, is_if_true)
            self.nextBlock(end)
            return
        elif isinstance(test, ast.Compare):
            if len(test.ops) > 1:
                cleanup = self.newBlock()
                self.visit(test.left)
                for (op, comparator) in zip(test.ops[:-1], test.comparators[:-1]):
                    self.emitChainedCompareStep(op, comparator, cleanup, always_pop=True)
                self.visit(test.comparators[-1])
                self.defaultEmitCompare(test.ops[-1])
                self.emit('POP_JUMP_IF_TRUE' if is_if_true else 'POP_JUMP_IF_FALSE', next)
                self.nextBlock()
                end = self.newBlock()
                self.emit_noline('JUMP_FORWARD', end)
                self.nextBlock(cleanup)
                self.emit('POP_TOP')
                if not is_if_true:
                    self.emit_noline('JUMP_FORWARD', next)
                self.nextBlock(end)
                return
        self.visit(test)
        self.emit('POP_JUMP_IF_TRUE' if is_if_true else 'POP_JUMP_IF_FALSE', next)
        self.nextBlock()

    def visitIfExp(self, node):
        if False:
            return 10
        endblock = self.newBlock()
        elseblock = self.newBlock()
        self.compileJumpIf(node.test, elseblock, False)
        self.visit(node.body)
        self.emit_noline('JUMP_FORWARD', endblock)
        self.nextBlock(elseblock)
        self.visit(node.orelse)
        self.nextBlock(endblock)

    def emitChainedCompareStep(self, op, value, cleanup, always_pop=False):
        if False:
            i = 10
            return i + 15
        self.visit(value)
        self.emit('DUP_TOP')
        self.emit('ROT_THREE')
        self.defaultEmitCompare(op)
        self.emit('POP_JUMP_IF_FALSE' if always_pop else 'JUMP_IF_FALSE_OR_POP', cleanup)
        self.nextBlock(label='compare_or_cleanup')

    def defaultEmitCompare(self, op):
        if False:
            return 10
        if isinstance(op, ast.Is):
            self.emit('IS_OP', 0)
        elif isinstance(op, ast.IsNot):
            self.emit('IS_OP', 1)
        elif isinstance(op, ast.In):
            self.emit('CONTAINS_OP', 0)
        elif isinstance(op, ast.NotIn):
            self.emit('CONTAINS_OP', 1)
        else:
            self.emit('COMPARE_OP', self._cmp_opcode[type(op)])

    def visitCompare(self, node):
        if False:
            while True:
                i = 10
        self.visit(node.left)
        cleanup = self.newBlock('cleanup')
        for (op, code) in zip(node.ops[:-1], node.comparators[:-1]):
            self.emitChainedCompareStep(op, code, cleanup)
        if node.ops:
            op = node.ops[-1]
            code = node.comparators[-1]
            self.visit(code)
            self.defaultEmitCompare(op)
        if len(node.ops) > 1:
            end = self.newBlock('end')
            self.emit('JUMP_FORWARD', end)
            self.nextBlock(cleanup)
            self.emit('ROT_TWO')
            self.emit('POP_TOP')
            self.nextBlock(end)

    def get_qual_prefix(self, gen):
        if False:
            return 10
        prefix = ''
        if gen.scope.global_scope:
            return prefix
        parent = gen.scope.parent
        while not isinstance(parent, symbols.ModuleScope):
            if parent.is_function_scope:
                prefix = parent.name + '.<locals>.' + prefix
            else:
                prefix = parent.name + '.' + prefix
            if parent.global_scope:
                break
            parent = parent.parent
        return prefix

    def _makeClosure(self, gen, flags):
        if False:
            for i in range(10):
                print('nop')
        prefix = ''
        if not isinstance(gen.tree, ast.ClassDef):
            prefix = self.get_qual_prefix(gen)
        frees = gen.scope.get_free_vars()
        if frees:
            for name in frees:
                self.emit('LOAD_CLOSURE', name)
            self.emit('BUILD_TUPLE', len(frees))
            flags |= 8
        gen.set_qual_name(prefix + gen.name)
        self.emit('LOAD_CONST', gen)
        self.emit('LOAD_CONST', prefix + gen.name)
        self.emit('MAKE_FUNCTION', flags)

    def visitDelete(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.targets)

    def conjure_arguments(self, args: List[ast.arg]) -> ast.arguments:
        if False:
            return 10
        return ast.arguments([], args, None, [], [], None, [])

    def compile_comprehension(self, node: CompNode, name: str, elt: ast.expr, val: ast.expr | None, opcode: str, oparg: object=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        is_async_function = self.scope.coroutine
        args = self.conjure_arguments([ast.arg('.0', None)])
        gen = self.make_func_codegen(node, args, name, node.lineno)
        gen.set_lineno(node)
        is_async_generator = gen.scope.coroutine
        if is_async_generator and (not is_async_function) and (not isinstance(node, ast.GeneratorExp)):
            raise self.syntax_error('asynchronous comprehension outside of an asynchronous function', node)
        if opcode:
            gen.emit(opcode, oparg)
        gen.compile_comprehension_generator(node.generators, 0, 0, elt, val, type(node), True)
        if not isinstance(node, ast.GeneratorExp):
            gen.emit('RETURN_VALUE')
        gen.finishFunction()
        self._makeClosure(gen, 0)
        self.visit(node.generators[0].iter)
        if node.generators[0].is_async:
            self.emit('GET_AITER')
        else:
            self.emit('GET_ITER')
        self.emit('CALL_FUNCTION', 1)
        if gen.scope.coroutine and type(node) is not ast.GeneratorExp:
            self.emit('GET_AWAITABLE')
            self.emit('LOAD_CONST', None)
            self.emit('YIELD_FROM')

    def visitGeneratorExp(self, node):
        if False:
            return 10
        self.compile_comprehension(node, sys.intern('<genexpr>'), node.elt, None, None)

    def visitListComp(self, node):
        if False:
            i = 10
            return i + 15
        self.compile_comprehension(node, sys.intern('<listcomp>'), node.elt, None, 'BUILD_LIST')

    def visitSetComp(self, node):
        if False:
            i = 10
            return i + 15
        self.compile_comprehension(node, sys.intern('<setcomp>'), node.elt, None, 'BUILD_SET')

    def visitDictComp(self, node):
        if False:
            i = 10
            return i + 15
        self.compile_comprehension(node, sys.intern('<dictcomp>'), node.key, node.value, 'BUILD_MAP')

    def compile_comprehension_generator(self, generators, gen_index, depth, elt, val, type, outermost_gen_is_param):
        if False:
            print('Hello World!')
        if generators[gen_index].is_async:
            self.compile_async_comprehension(generators, gen_index, depth, elt, val, type, outermost_gen_is_param)
        else:
            self.compile_sync_comprehension(generators, gen_index, depth, elt, val, type, outermost_gen_is_param)

    def compile_async_comprehension(self, generators, gen_index, depth, elt, val, type, outermost_gen_is_param):
        if False:
            i = 10
            return i + 15
        start = self.newBlock('start')
        except_ = self.newBlock('except')
        if_cleanup = self.newBlock('if_cleanup')
        gen = generators[gen_index]
        if gen_index == 0 and outermost_gen_is_param:
            self.loadName('.0')
        else:
            self.visit(gen.iter)
            self.emit('GET_AITER')
        self.nextBlock(start)
        self.emit('SETUP_FINALLY', except_)
        self.emit('GET_ANEXT')
        self.emit('LOAD_CONST', None)
        self.emit('YIELD_FROM')
        self.emit('POP_BLOCK')
        self.visit(gen.target)
        for if_ in gen.ifs:
            self.compileJumpIf(if_, if_cleanup, False)
            self.newBlock()
        depth += 1
        gen_index += 1
        if gen_index < len(generators):
            self.compile_comprehension_generator(generators, gen_index, depth, elt, val, type, False)
        elif type is ast.GeneratorExp:
            self.visit(elt)
            self.emit('YIELD_VALUE')
            self.emit('POP_TOP')
        elif type is ast.ListComp:
            self.visit(elt)
            self.emit('LIST_APPEND', depth + 1)
        elif type is ast.SetComp:
            self.visit(elt)
            self.emit('SET_ADD', depth + 1)
        elif type is ast.DictComp:
            self.compile_dictcomp_element(elt, val)
            self.emit('MAP_ADD', depth + 1)
        else:
            raise NotImplementedError('unknown comprehension type')
        self.nextBlock(if_cleanup)
        self.emit('JUMP_ABSOLUTE', start)
        self.nextBlock(except_)
        self.emit('END_ASYNC_FOR')

    def compile_sync_comprehension(self, generators, gen_index, depth, elt, val, type, outermost_gen_is_param):
        if False:
            return 10
        start = self.newBlock('start')
        skip = self.newBlock('skip')
        if_cleanup = self.newBlock('if_cleanup')
        anchor = self.newBlock('anchor')
        gen = generators[gen_index]
        if gen_index == 0 and outermost_gen_is_param:
            self.loadName('.0')
        else:
            if isinstance(gen.iter, (ast.Tuple, ast.List)):
                elts = gen.iter.elts
                if len(elts) == 1 and (not isinstance(elts[0], ast.Starred)):
                    self.visit(elts[0])
                    start = None
            if start:
                self.visit(gen.iter)
                self.emit('GET_ITER')
        if start:
            depth += 1
            self.nextBlock(start)
            self.emit('FOR_ITER', anchor)
            self.nextBlock()
        self.visit(gen.target)
        for if_ in gen.ifs:
            self.compileJumpIf(if_, if_cleanup, False)
            self.newBlock()
        gen_index += 1
        if gen_index < len(generators):
            self.compile_comprehension_generator(generators, gen_index, depth, elt, val, type, False)
        else:
            if type is ast.GeneratorExp:
                self.visit(elt)
                self.emit('YIELD_VALUE')
                self.emit('POP_TOP')
            elif type is ast.ListComp:
                self.visit(elt)
                self.emit('LIST_APPEND', depth + 1)
            elif type is ast.SetComp:
                self.visit(elt)
                self.emit('SET_ADD', depth + 1)
            elif type is ast.DictComp:
                self.compile_dictcomp_element(elt, val)
                self.emit('MAP_ADD', depth + 1)
            else:
                raise NotImplementedError('unknown comprehension type')
            self.nextBlock(skip)
        self.nextBlock(if_cleanup)
        if start:
            self.emit('JUMP_ABSOLUTE', start)
            self.nextBlock(anchor)

    def compile_dictcomp_element(self, elt, val):
        if False:
            for i in range(10):
                print('nop')
        self.visit(elt)
        self.visit(val)

    def visitAssert(self, node):
        if False:
            print('Hello World!')
        if not self.optimization_lvl:
            end = self.newBlock()
            self.compileJumpIf(node.test, end, True)
            self.emit('LOAD_ASSERTION_ERROR')
            if node.msg:
                self.visit(node.msg)
                self.emit('CALL_FUNCTION', 1)
                self.emit('RAISE_VARARGS', 1)
            else:
                self.emit('RAISE_VARARGS', 1)
            self.nextBlock(end)

    def visitRaise(self, node):
        if False:
            for i in range(10):
                print('nop')
        n = 0
        if node.exc:
            self.visit(node.exc)
            n = n + 1
        if node.cause:
            self.visit(node.cause)
            n = n + 1
        self.emit('RAISE_VARARGS', n)
        self.nextBlock()

    def visitTry(self, node):
        if False:
            i = 10
            return i + 15
        if node.finalbody:
            if node.handlers:
                self.emit_try_finally(node, lambda : self.visitTryExcept(node), lambda : self.visitStatements(node.finalbody))
            else:
                self.emit_try_finally(node, lambda : self.visitStatements(node.body), lambda : self.visitStatements(node.finalbody))
            return
        self.visitTryExcept(node)

    def visitTryExcept(self, node):
        if False:
            return 10
        body = self.newBlock('try_body')
        except_ = self.newBlock('try_handlers')
        orElse = self.newBlock('try_else')
        end = self.newBlock('try_end')
        self.emit('SETUP_FINALLY', except_)
        self.nextBlock(body)
        self.setups.push(Entry(TRY_EXCEPT, body, None, None))
        self.visitStatements(node.body)
        self.setups.pop()
        self.set_no_lineno()
        self.emit('POP_BLOCK')
        self.emit('JUMP_FORWARD', orElse)
        self.nextBlock(except_)
        self.setups.push(Entry(EXCEPTION_HANDLER, None, None, None))
        last = len(node.handlers) - 1
        for i in range(len(node.handlers)):
            handler = node.handlers[i]
            expr = handler.type
            target = handler.name
            body = handler.body
            self.set_lineno(handler)
            except_ = self.newBlock(f'try_except_{i}')
            if expr:
                self.emit('DUP_TOP')
                self.visit(expr)
                self.emit('JUMP_IF_NOT_EXC_MATCH', except_)
                self.nextBlock()
            elif i < last:
                raise SyntaxError("default 'except:' must be last", self.syntax_error_position(handler))
            else:
                self.set_lineno(handler)
            self.emit('POP_TOP')
            if target:
                cleanup_end = self.newBlock(f'try_cleanup_end{i}')
                cleanup_body = self.newBlock(f'try_cleanup_body{i}')
                self.storeName(target)
                self.emit('POP_TOP')
                self.emit('SETUP_FINALLY', cleanup_end)
                self.nextBlock(cleanup_body)
                self.setups.push(Entry(HANDLER_CLEANUP, cleanup_body, cleanup_end, target))
                self.visit(body)
                self.setups.pop()
                self.set_no_lineno()
                self.emit('POP_BLOCK')
                self.emit('POP_EXCEPT')
                self.emit('LOAD_CONST', None)
                self.storeName(target)
                self.delName(target)
                self.emit('JUMP_FORWARD', end)
                self.nextBlock(cleanup_end)
                self.set_no_lineno()
                self.emit('LOAD_CONST', None)
                self.storeName(target)
                self.delName(target)
                self.emit('RERAISE', 1)
            else:
                cleanup_body = self.newBlock(f'try_cleanup_body{i}')
                self.emit('POP_TOP')
                self.emit('POP_TOP')
                self.nextBlock(cleanup_body)
                self.setups.push(Entry(HANDLER_CLEANUP, cleanup_body, None, None))
                self.visit(body)
                self.setups.pop()
                self.set_no_lineno()
                self.emit('POP_EXCEPT')
                self.emit('JUMP_FORWARD', end)
            self.nextBlock(except_)
        self.setups.pop()
        self.set_no_lineno()
        self.emit('RERAISE', 0)
        self.nextBlock(orElse)
        self.visitStatements(node.orelse)
        self.nextBlock(end)

    def emit_except_local(self, handler: ast.ExceptHandler):
        if False:
            print('Hello World!')
        target = handler.name
        type_ = handler.type
        if target:
            if type_:
                self.set_lineno(type_)
            self.storeName(target)
        else:
            self.emit('POP_TOP')
        self.emit('POP_TOP')

    def emit_try_finally(self, node, try_body, finalbody, except_protect=False):
        if False:
            return 10
        '\n        The overall idea is:\n           SETUP_FINALLY end\n           try-body\n           POP_BLOCK\n           finally-body\n           JUMP exit\n        end:\n           finally-body\n        exit:\n        '
        body = self.newBlock('try_finally_body')
        end = self.newBlock('try_finally_end')
        exit_ = self.newBlock('try_finally_exit')
        self.emit('SETUP_FINALLY', end)
        self.nextBlock(body)
        self.setups.push(Entry(FINALLY_TRY, body, end, finalbody))
        try_body()
        self.emit_noline('POP_BLOCK')
        self.setups.pop()
        finalbody()
        self.emit_noline('JUMP_FORWARD', exit_)
        self.nextBlock(end)
        self.setups.push(Entry(FINALLY_END, end, None, None))
        finalbody()
        self.setups.pop()
        self.emit('RERAISE', 0)
        self.nextBlock(exit_)

    def call_exit_with_nones(self):
        if False:
            print('Hello World!')
        self.emit('LOAD_CONST', None)
        self.emit('DUP_TOP')
        self.emit('DUP_TOP')
        self.emit('CALL_FUNCTION', 3)

    def visitWith_(self, node, kind, pos=0):
        if False:
            while True:
                i = 10
        item = node.items[pos]
        block = self.newBlock('with_block')
        finally_ = self.newBlock('with_finally')
        exit_ = self.newBlock('with_exit')
        self.visit(item.context_expr)
        if kind == ASYNC_WITH:
            self.emit('BEFORE_ASYNC_WITH')
            self.emit('GET_AWAITABLE')
            self.emit('LOAD_CONST', None)
            self.emit('YIELD_FROM')
            self.emit('SETUP_ASYNC_WITH', finally_)
        else:
            self.emit('SETUP_WITH', finally_)
        self.nextBlock(block)
        self.setups.push(Entry(kind, block, finally_, node))
        if item.optional_vars:
            self.visit(item.optional_vars)
        else:
            self.emit('POP_TOP')
        if pos + 1 < len(node.items):
            self.visitWith_(node, kind, pos + 1)
        else:
            self.visitStatements(node.body)
        if kind == WITH:
            self.set_no_lineno()
        self.setups.pop()
        self.emit('POP_BLOCK')
        self.set_lineno(node)
        self.call_exit_with_nones()
        if kind == ASYNC_WITH:
            self.emit('GET_AWAITABLE')
            self.emit('LOAD_CONST', None)
            self.emit('YIELD_FROM')
        self.emit('POP_TOP')
        if kind == ASYNC_WITH:
            self.emit('JUMP_ABSOLUTE', exit_)
        else:
            self.emit('JUMP_FORWARD', exit_)
        self.nextBlock(finally_)
        self.emit('WITH_EXCEPT_START')
        if kind == ASYNC_WITH:
            self.emit('GET_AWAITABLE')
            self.emit('LOAD_CONST', None)
            self.emit('YIELD_FROM')
        except_ = self.newBlock()
        self.emit('POP_JUMP_IF_TRUE', except_)
        self.nextBlock()
        self.emit('RERAISE', 1)
        self.nextBlock(except_)
        self.emit('POP_TOP')
        self.emit('POP_TOP')
        self.emit('POP_TOP')
        self.emit('POP_EXCEPT')
        self.emit('POP_TOP')
        self.nextBlock(exit_)

    def visitWith(self, node):
        if False:
            print('Hello World!')
        self.visitWith_(node, WITH, 0)

    def visitAsyncWith(self, node, pos=0):
        if False:
            while True:
                i = 10
        if not self.scope.coroutine:
            raise SyntaxError("'async with' outside async function", self.syntax_error_position(node))
        self.visitWith_(node, ASYNC_WITH, 0)

    def visitExpr(self, node):
        if False:
            while True:
                i = 10
        if self.interactive:
            self.visit(node.value)
            self.emit('PRINT_EXPR')
        elif is_const(node.value):
            self.emit('NOP')
        else:
            self.visit(node.value)
            self.set_no_lineno()
            self.emit('POP_TOP')

    def visitConstant(self, node: ast.Constant):
        if False:
            for i in range(10):
                print('nop')
        self.emit('LOAD_CONST', node.value)

    def visitKeyword(self, node):
        if False:
            i = 10
            return i + 15
        self.emit('LOAD_CONST', node.name)
        self.visit(node.expr)

    def visitGlobal(self, node):
        if False:
            i = 10
            return i + 15
        pass

    def visitNonlocal(self, node):
        if False:
            while True:
                i = 10
        pass

    def visitName(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.ctx, ast.Store):
            self.storeName(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.delName(node.id)
        else:
            self.loadName(node.id)

    def visitPass(self, node):
        if False:
            while True:
                i = 10
        self.emit('NOP')

    def visitImport(self, node):
        if False:
            return 10
        level = 0
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            self.emit('LOAD_CONST', level)
            self.emit('LOAD_CONST', None)
            self.emit('IMPORT_NAME', self.mangle(name))
            mod = name.split('.')[0]
            if asname:
                self.emitImportAs(name, asname)
            else:
                self.storeName(mod)

    def visitImportFrom(self, node):
        if False:
            return 10
        level = node.level
        fromlist = tuple((alias.name for alias in node.names))
        self.emit('LOAD_CONST', level)
        self.emit('LOAD_CONST', fromlist)
        self.emit('IMPORT_NAME', node.module or '')
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            if name == '*':
                self.namespace = 0
                self.emit('IMPORT_STAR')
                assert len(node.names) == 1
                return
            else:
                self.emit('IMPORT_FROM', name)
                self.storeName(asname or name)
        self.emit('POP_TOP')

    def emitImportAs(self, name: str, asname: str):
        if False:
            for i in range(10):
                print('nop')
        elts = name.split('.')
        if len(elts) == 1:
            self.storeName(asname)
            return
        first = True
        for elt in elts[1:]:
            if not first:
                self.emit('ROT_TWO')
                self.emit('POP_TOP')
            self.emit('IMPORT_FROM', elt)
            first = False
        self.storeName(asname)
        self.emit('POP_TOP')

    def visitAttribute(self, node):
        if False:
            while True:
                i = 10
        self.visit(node.value)
        if isinstance(node.ctx, ast.Store):
            with self.temp_lineno(node.end_lineno):
                self.emit('STORE_ATTR', self.mangle(node.attr))
        elif isinstance(node.ctx, ast.Del):
            self.emit('DELETE_ATTR', self.mangle(node.attr))
        else:
            with self.temp_lineno(node.end_lineno):
                self.emit('LOAD_ATTR', self.mangle(node.attr))

    def visitAssign(self, node):
        if False:
            print('Hello World!')
        self.visit(node.value)
        dups = len(node.targets) - 1
        for i in range(len(node.targets)):
            elt = node.targets[i]
            if i < dups:
                self.emit('DUP_TOP')
            if isinstance(elt, ast.AST):
                self.visit(elt)

    def checkAnnExpr(self, node):
        if False:
            i = 10
            return i + 15
        self.visit(node)
        self.emit('POP_TOP')

    def checkAnnSlice(self, node):
        if False:
            print('Hello World!')
        if node.lower:
            self.checkAnnExpr(node.lower)
        if node.upper:
            self.checkAnnExpr(node.upper)
        if node.step:
            self.checkAnnExpr(node.step)

    def checkAnnSubscr(self, node):
        if False:
            print('Hello World!')
        if isinstance(node, ast.Slice):
            self.checkAnnSlice(node)
        elif isinstance(node, ast.Tuple):
            for v in node.elts:
                self.checkAnnSubscr(v)
        else:
            self.checkAnnExpr(node)

    def checkAnnotation(self, node):
        if False:
            print('Hello World!')
        if isinstance(self.tree, (ast.Module, ast.ClassDef)):
            if self.future_flags & consts.CO_FUTURE_ANNOTATIONS:
                with self.noEmit():
                    self._visitAnnotation(node.annotation)
            else:
                self.checkAnnExpr(node.annotation)

    def findAnn(self, stmts):
        if False:
            print('Hello World!')
        for stmt in stmts:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            elif isinstance(stmt, ast.AnnAssign):
                return True
            elif isinstance(stmt, (ast.stmt, ast.ExceptHandler)):
                for field in stmt._fields:
                    child = getattr(stmt, field)
                    if isinstance(child, list):
                        if self.findAnn(child):
                            return True
        return False

    def emitStoreAnnotation(self, name: str, annotation: ast.expr):
        if False:
            i = 10
            return i + 15
        assert self.did_setup_annotations
        self._visitAnnotation(annotation)
        self.emit('LOAD_NAME', '__annotations__')
        mangled = self.mangle(name)
        self.emit('LOAD_CONST', mangled)
        self.emit('STORE_SUBSCR')

    def visitAnnAssign(self, node):
        if False:
            return 10
        if node.value:
            self.visit(node.value)
            self.visit(node.target)
        if isinstance(node.target, ast.Name):
            if node.simple and isinstance(self.tree, (ast.Module, ast.ClassDef)):
                self.emitStoreAnnotation(node.target.id, node.annotation)
            else:
                with self.noEmit():
                    self._visitAnnotation(node.annotation)
        elif isinstance(node.target, ast.Attribute):
            if not node.value:
                self.checkAnnExpr(node.target.value)
        elif isinstance(node.target, ast.Subscript):
            if not node.value:
                self.checkAnnExpr(node.target.value)
                self.checkAnnSubscr(node.target.slice)
        else:
            raise SystemError(f'invalid node type {type(node).__name__} for annotated assignment')
        if not node.simple:
            self.checkAnnotation(node)

    def visitAssName(self, node):
        if False:
            while True:
                i = 10
        if node.flags == 'OP_ASSIGN':
            self.storeName(node.name)
        elif node.flags == 'OP_DELETE':
            self.set_lineno(node)
            self.delName(node.name)
        else:
            print('oops', node.flags)
            assert 0

    def visitAssAttr(self, node):
        if False:
            print('Hello World!')
        self.visit(node.expr)
        if node.flags == 'OP_ASSIGN':
            self.emit('STORE_ATTR', self.mangle(node.attrname))
        elif node.flags == 'OP_DELETE':
            self.emit('DELETE_ATTR', self.mangle(node.attrname))
        else:
            print('warning: unexpected flags:', node.flags)
            print(node)
            assert 0

    def _visitAssSequence(self, node, op='UNPACK_SEQUENCE'):
        if False:
            for i in range(10):
                print('nop')
        if findOp(node) != 'OP_DELETE':
            self.emit(op, len(node.nodes))
        for child in node.nodes:
            self.visit(child)
    visitAssTuple = _visitAssSequence
    visitAssList = _visitAssSequence

    def visitAugAssign(self, node):
        if False:
            i = 10
            return i + 15
        self.set_lineno(node.target)
        if isinstance(node.target, ast.Attribute):
            self.emitAugAttribute(node)
        elif isinstance(node.target, ast.Name):
            self.emitAugName(node)
        else:
            self.emitAugSubscript(node)
    _augmented_opcode = {ast.Add: 'INPLACE_ADD', ast.Sub: 'INPLACE_SUBTRACT', ast.Mult: 'INPLACE_MULTIPLY', ast.MatMult: 'INPLACE_MATRIX_MULTIPLY', ast.Div: 'INPLACE_TRUE_DIVIDE', ast.FloorDiv: 'INPLACE_FLOOR_DIVIDE', ast.Mod: 'INPLACE_MODULO', ast.Pow: 'INPLACE_POWER', ast.RShift: 'INPLACE_RSHIFT', ast.LShift: 'INPLACE_LSHIFT', ast.BitAnd: 'INPLACE_AND', ast.BitXor: 'INPLACE_XOR', ast.BitOr: 'INPLACE_OR'}

    def emitAugRHS(self, node):
        if False:
            print('Hello World!')
        with self.temp_lineno(node.lineno):
            self.visit(node.value)
            self.emit(self._augmented_opcode[type(node.op)])

    def emitAugName(self, node):
        if False:
            return 10
        target = node.target
        self.loadName(target.id)
        self.emitAugRHS(node)
        self.set_lineno(target)
        self.storeName(target.id)

    def emitAugAttribute(self, node):
        if False:
            return 10
        target = node.target
        self.visit(target.value)
        self.emit('DUP_TOP')
        with self.temp_lineno(node.target.end_lineno):
            self.emit('LOAD_ATTR', self.mangle(target.attr))
        self.emitAugRHS(node)
        self.graph.set_lineno(node.target.end_lineno)
        self.emit('ROT_TWO')
        self.emit('STORE_ATTR', self.mangle(target.attr))

    def emitAugSubscript(self, node):
        if False:
            while True:
                i = 10
        self.visitSubscript(node.target, 1)
        self.emitAugRHS(node)
        self.emit('ROT_THREE')
        self.emit('STORE_SUBSCR')

    def visitExec(self, node):
        if False:
            return 10
        self.visit(node.expr)
        if node.locals is None:
            self.emit('LOAD_CONST', None)
        else:
            self.visit(node.locals)
        if node.globals is None:
            self.emit('DUP_TOP')
        else:
            self.visit(node.globals)
        self.emit('EXEC_STMT')

    def compiler_subkwargs(self, kwargs, begin, end):
        if False:
            while True:
                i = 10
        nkwargs = end - begin
        big = nkwargs * 2 > STACK_USE_GUIDELINE
        if nkwargs > 1 and (not big):
            for i in range(begin, end):
                self.visit(kwargs[i].value)
            self.emit('LOAD_CONST', tuple((arg.arg for arg in kwargs[begin:end])))
            self.emit('BUILD_CONST_KEY_MAP', nkwargs)
            return
        if big:
            self.emit_noline('BUILD_MAP', 0)
        for i in range(begin, end):
            self.emit('LOAD_CONST', kwargs[i].arg)
            self.visit(kwargs[i].value)
            if big:
                self.emit_noline('MAP_ADD', 1)
        if not big:
            self.emit('BUILD_MAP', nkwargs)

    def _fastcall_helper(self, argcnt, node, args, kwargs):
        if False:
            return 10
        for arg in args:
            self.visit(arg)
        if len(kwargs) > 0:
            self.visit(kwargs)
            self.emit('LOAD_CONST', tuple((arg.arg for arg in kwargs)))
            self.emit('CALL_FUNCTION_KW', argcnt + len(args) + len(kwargs))
            return
        self.emit('CALL_FUNCTION', argcnt + len(args))

    def _call_helper(self, argcnt, node, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        starred = any((isinstance(arg, ast.Starred) for arg in args))
        mustdictunpack = any((arg.arg is None for arg in kwargs))
        manyargs = len(args) + len(kwargs) * 2 > STACK_USE_GUIDELINE
        if not (starred or mustdictunpack or manyargs):
            return self._fastcall_helper(argcnt, node, args, kwargs)
        if argcnt == 0 and len(args) == 1 and starred:
            self.visit(args[0].value)
        else:
            self._visitSequenceLoad(args, 'BUILD_LIST', 'LIST_APPEND', 'LIST_EXTEND', num_pushed=argcnt, is_tuple=True)
        nkwelts = len(kwargs)
        if nkwelts > 0:
            seen = 0
            have_dict = False
            for (i, arg) in enumerate(kwargs):
                if arg.arg is None:
                    if seen > 0:
                        self.compiler_subkwargs(kwargs, i - seen, i)
                        if have_dict:
                            self.emit('DICT_MERGE', 1)
                        have_dict = True
                        seen = 0
                    if not have_dict:
                        self.emit('BUILD_MAP', 0)
                        have_dict = True
                    self.visit(arg.value)
                    self.emit('DICT_MERGE', 1)
                else:
                    seen += 1
            if seen > 0:
                self.compiler_subkwargs(kwargs, nkwelts - seen, nkwelts)
                if have_dict:
                    self.emit('DICT_MERGE', 1)
        self.emit('CALL_FUNCTION_EX', int(nkwelts > 0))

    def visitCall(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.keywords or not isinstance(node.func, ast.Attribute) or (not isinstance(node.func.ctx, ast.Load)) or any((isinstance(arg, ast.Starred) for arg in node.args)) or (len(node.args) >= STACK_USE_GUIDELINE):
            self.visit(node.func)
            self._call_helper(0, node, node.args, node.keywords)
            return
        self.visit(node.func.value)
        with self.temp_lineno(node.func.end_lineno):
            self.emit('LOAD_METHOD', self.mangle(node.func.attr))
            for arg in node.args:
                self.visit(arg)
            nargs = len(node.args)
            self.emit('CALL_METHOD', nargs)

    def checkReturn(self, node):
        if False:
            while True:
                i = 10
        if not isinstance(self.tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise SyntaxError("'return' outside function", self.syntax_error_position(node))
        elif self.scope.coroutine and self.scope.generator and node.value:
            raise SyntaxError("'return' with value in async generator", self.syntax_error_position(node))

    def visitReturn(self, node):
        if False:
            print('Hello World!')
        self.checkReturn(node)
        preserve_tos = bool(node.value and (not isinstance(node.value, ast.Constant)))
        if preserve_tos:
            self.visit(node.value)
        elif node.value:
            self.set_lineno(node.value)
            self.emit('NOP')
        if not node.value or node.value.lineno != node.lineno:
            self.set_lineno(node)
            self.emit('NOP')
        self.unwind_setup_entries(preserve_tos)
        if not node.value:
            self.emit('LOAD_CONST', None)
        elif not preserve_tos:
            self.emit('LOAD_CONST', node.value.value)
        self.emit('RETURN_VALUE')
        self.nextBlock()

    def visitYield(self, node):
        if False:
            return 10
        if not isinstance(self.tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.GeneratorExp)):
            raise SyntaxError("'yield' outside function", self.syntax_error_position(node))
        if node.value:
            self.visit(node.value)
        else:
            self.emit('LOAD_CONST', None)
        self.emit('YIELD_VALUE')

    def visitYieldFrom(self, node):
        if False:
            print('Hello World!')
        if not isinstance(self.tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.GeneratorExp)):
            raise SyntaxError("'yield' outside function", self.syntax_error_position(node))
        elif self.scope.coroutine:
            raise SyntaxError("'yield from' inside async function", self.syntax_error_position(node))
        self.visit(node.value)
        self.emit('GET_YIELD_FROM_ITER')
        self.emit('LOAD_CONST', None)
        self.emit('YIELD_FROM')

    def visitAwait(self, node):
        if False:
            while True:
                i = 10
        self.visit(node.value)
        self.emit('GET_AWAITABLE')
        self.emit('LOAD_CONST', None)
        self.emit('YIELD_FROM')

    def visitSubscript(self, node, aug_flag=None):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.ctx, ast.Load):
            self.emit('BINARY_SUBSCR')
        elif isinstance(node.ctx, ast.Store):
            if aug_flag:
                self.emit('DUP_TOP_TWO')
                self.emit('BINARY_SUBSCR')
            else:
                self.emit('STORE_SUBSCR')
        elif isinstance(node.ctx, ast.Del):
            self.emit('DELETE_SUBSCR')
        else:
            assert 0

    def binaryOp(self, node, op):
        if False:
            print('Hello World!')
        self.visit(node.left)
        self.visit(node.right)
        self.emit(op)
    _binary_opcode: Dict[Type, str] = {ast.Add: 'BINARY_ADD', ast.Sub: 'BINARY_SUBTRACT', ast.Mult: 'BINARY_MULTIPLY', ast.MatMult: 'BINARY_MATRIX_MULTIPLY', ast.Div: 'BINARY_TRUE_DIVIDE', ast.FloorDiv: 'BINARY_FLOOR_DIVIDE', ast.Mod: 'BINARY_MODULO', ast.Pow: 'BINARY_POWER', ast.LShift: 'BINARY_LSHIFT', ast.RShift: 'BINARY_RSHIFT', ast.BitOr: 'BINARY_OR', ast.BitXor: 'BINARY_XOR', ast.BitAnd: 'BINARY_AND'}

    def visitBinOp(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.left)
        self.visit(node.right)
        op = self._binary_opcode[type(node.op)]
        self.emit(op)

    def unaryOp(self, node, op):
        if False:
            print('Hello World!')
        self.visit(node.operand)
        self.emit(op)
    _unary_opcode: Dict[Type, str] = {ast.Invert: 'UNARY_INVERT', ast.USub: 'UNARY_NEGATIVE', ast.UAdd: 'UNARY_POSITIVE', ast.Not: 'UNARY_NOT'}

    def visitUnaryOp(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.unaryOp(node, self._unary_opcode[type(node.op)])

    def visitBackquote(self, node):
        if False:
            i = 10
            return i + 15
        return self.unaryOp(node, 'UNARY_CONVERT')

    def visitEllipsis(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.emit('LOAD_CONST', Ellipsis)

    def _visitUnpack(self, node):
        if False:
            i = 10
            return i + 15
        before = 0
        after = 0
        starred = None
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                if starred is not None:
                    raise SyntaxError('multiple starred expressions in assignment', self.syntax_error_position(elt))
                elif before >= 256 or len(node.elts) - before - 1 >= 1 << 31 >> 8:
                    raise SyntaxError('too many expressions in star-unpacking assignment', self.syntax_error_position(elt))
                starred = elt.value
            elif starred:
                after += 1
            else:
                before += 1
        if starred:
            self.emit('UNPACK_EX', after << 8 | before)
        else:
            self.emit('UNPACK_SEQUENCE', before)

    def hasStarred(self, elts):
        if False:
            while True:
                i = 10
        for elt in elts:
            if isinstance(elt, ast.Starred):
                return True
        return False

    def _visitSequenceLoad(self, elts, build_op, add_op, extend_op, num_pushed=0, is_tuple=False):
        if False:
            print('Hello World!')
        if len(elts) > 2 and all((isinstance(elt, ast.Constant) for elt in elts)):
            elts_tuple = tuple((elt.value for elt in elts))
            if is_tuple:
                self.emit('LOAD_CONST', elts_tuple)
            else:
                if add_op == 'SET_ADD':
                    elts_tuple = frozenset(elts_tuple)
                self.emit(build_op, num_pushed)
                self.emit('LOAD_CONST', elts_tuple)
                self.emit(extend_op, 1)
            return
        big = len(elts) + num_pushed > STACK_USE_GUIDELINE
        starred_load = self.hasStarred(elts)
        if not starred_load and (not big):
            for elt in elts:
                self.visit(elt)
            collection_size = num_pushed + len(elts)
            self.emit('BUILD_TUPLE' if is_tuple else build_op, collection_size)
            return
        sequence_built = False
        if big:
            self.emit(build_op, num_pushed)
            sequence_built = True
        on_stack = 0
        for elt in elts:
            if isinstance(elt, ast.Starred):
                if not sequence_built:
                    self.emit(build_op, on_stack + num_pushed)
                    sequence_built = True
                self.visit(elt.value)
                self.emit(extend_op, 1)
            else:
                self.visit(elt)
                if sequence_built:
                    self.emit(add_op, 1)
                else:
                    on_stack += 1
        if is_tuple:
            self.emit('LIST_TO_TUPLE')

    def _visitSequence(self, node, build_op, add_op, extend_op, ctx, is_tuple=False):
        if False:
            return 10
        if isinstance(ctx, ast.Store):
            self._visitUnpack(node)
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    self.visit(elt.value)
                else:
                    self.visit(elt)
            return
        elif isinstance(ctx, ast.Load):
            return self._visitSequenceLoad(node.elts, build_op, add_op, extend_op, num_pushed=0, is_tuple=is_tuple)
        else:
            return self.visit(node.elts)

    def visitStarred(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.ctx, ast.Store):
            raise SyntaxError('starred assignment target must be in a list or tuple', self.syntax_error_position(node))
        else:
            raise SyntaxError("can't use starred expression here", self.syntax_error_position(node))

    def visitTuple(self, node):
        if False:
            return 10
        self._visitSequence(node, 'BUILD_LIST', 'LIST_APPEND', 'LIST_EXTEND', node.ctx, is_tuple=True)

    def visitList(self, node):
        if False:
            return 10
        self._visitSequence(node, 'BUILD_LIST', 'LIST_APPEND', 'LIST_EXTEND', node.ctx)

    def visitSet(self, node):
        if False:
            return 10
        self._visitSequence(node, 'BUILD_SET', 'SET_ADD', 'SET_UPDATE', ast.Load())

    def visitSlice(self, node):
        if False:
            print('Hello World!')
        num = 2
        if node.lower:
            self.visit(node.lower)
        else:
            self.emit('LOAD_CONST', None)
        if node.upper:
            self.visit(node.upper)
        else:
            self.emit('LOAD_CONST', None)
        if node.step:
            self.visit(node.step)
            num += 1
        self.emit('BUILD_SLICE', num)

    def visitExtSlice(self, node):
        if False:
            while True:
                i = 10
        for d in node.dims:
            self.visit(d)
        self.emit('BUILD_TUPLE', len(node.dims))

    def visitNamedExpr(self, node: ast.NamedExpr):
        if False:
            for i in range(10):
                print('nop')
        self.visit(node.value)
        self.emit('DUP_TOP')
        self.visit(node.target)

    def _const_value(self, node):
        if False:
            i = 10
            return i + 15
        assert isinstance(node, ast.Constant)
        return node.value

    def get_bool_const(self, node) -> bool | None:
        if False:
            while True:
                i = 10
        'Return True if node represent constantly true value, False if\n        constantly false value, and None otherwise (non-constant).'
        if isinstance(node, ast.Constant):
            return bool(node.value)

    def compile_subdict(self, node, begin, end):
        if False:
            while True:
                i = 10
        n = end - begin
        big = n * 2 > STACK_USE_GUIDELINE
        if n > 1 and (not big) and all_items_const(node.keys, begin, end):
            for i in range(begin, end):
                self.visit(node.values[i])
            self.emit('LOAD_CONST', tuple((self._const_value(x) for x in node.keys[begin:end])))
            self.emit('BUILD_CONST_KEY_MAP', n)
            return
        if big:
            self.emit('BUILD_MAP', 0)
        for i in range(begin, end):
            self.visit(node.keys[i])
            self.visit(node.values[i])
            if big:
                self.emit('MAP_ADD', 1)
        if not big:
            self.emit('BUILD_MAP', n)

    def visitDict(self, node):
        if False:
            return 10
        elements = 0
        is_unpacking = False
        have_dict = False
        n = len(node.values)
        for (i, (k, v)) in enumerate(zip(node.keys, node.values)):
            is_unpacking = k is None
            if is_unpacking:
                if elements:
                    self.compile_subdict(node, i - elements, i)
                    if have_dict:
                        self.emit('DICT_UPDATE', 1)
                    have_dict = True
                    elements = 0
                if not have_dict:
                    self.emit('BUILD_MAP', 0)
                    have_dict = True
                self.visit(v)
                self.emit('DICT_UPDATE', 1)
            elif elements * 2 > STACK_USE_GUIDELINE:
                self.compile_subdict(node, i - elements, i + 1)
                if have_dict:
                    self.emit('DICT_UPDATE', 1)
                have_dict = True
                elements = 0
            else:
                elements += 1
        if elements:
            self.compile_subdict(node, n - elements, n)
            if have_dict:
                self.emit('DICT_UPDATE', 1)
            have_dict = True
        if not have_dict:
            self.emit('BUILD_MAP')

    def visitMatch(self, node: ast.Match) -> None:
        if False:
            for i in range(10):
                print('nop')
        'See compiler_match_inner in compile.c'
        pc = PatternContext()
        self.visit(node.subject)
        end = self.newBlock('match_end')
        assert node.cases, node.cases
        last_case = node.cases[-1]
        has_default = self._wildcard_check(node.cases[-1].pattern) and len(node.cases) > 1
        cases = list(node.cases)
        if has_default:
            cases.pop()
        for case in cases:
            self.set_lineno(case.pattern)
            is_last_non_default_case = case is cases[-1]
            if not is_last_non_default_case:
                self.emit('DUP_TOP')
            pc.stores = []
            pc.allow_irrefutable = case.guard is not None or case is last_case
            pc.fail_pop = []
            pc.on_top = 0
            self.visit(case.pattern, pc)
            assert not pc.on_top
            for name in pc.stores:
                self.storeName(name)
            guard = case.guard
            if guard:
                self._ensure_fail_pop(pc, 0)
                self.compileJumpIf(guard, pc.fail_pop[0], False)
            if not is_last_non_default_case:
                self.emit('POP_TOP')
            self.visit(case.body)
            self.emit('JUMP_FORWARD', end)
            self.set_lineno(case.pattern)
            self._emit_and_reset_fail_pop(pc)
        if has_default:
            self.set_lineno(last_case.pattern)
            if len(node.cases) == 1:
                self.emit('POP_TOP')
            else:
                self.emit('NOP')
            if last_case.guard:
                self.compileJumpIf(last_case.guard, end, False)
            self.visit(last_case.body)
        self.nextBlock(end)

    def visitMatchValue(self, node: ast.MatchValue, pc: PatternContext) -> None:
        if False:
            print('Hello World!')
        'See compiler_pattern_value in compile.c'
        value = node.value
        if not isinstance(value, ast.Constant | ast.Attribute):
            raise self.syntax_error('patterns may only match literals and attribute lookups', node)
        self.visit(value)
        self.emit('COMPARE_OP', '==')
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')

    def visitMatchSingleton(self, node: ast.MatchSingleton, pc: PatternContext) -> None:
        if False:
            return 10
        self.emit('LOAD_CONST', node.value)
        self.emit('IS_OP')
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')

    def visitMatchSequence(self, node: ast.MatchSequence, pc: PatternContext) -> None:
        if False:
            print('Hello World!')
        'See compiler_pattern_sequence in compile.c'
        patterns = node.patterns
        size = len(patterns)
        star = -1
        only_wildcard = True
        star_wildcard = False
        for (i, pattern) in enumerate(patterns):
            if isinstance(pattern, ast.MatchStar):
                if star >= 0:
                    raise self.syntax_error('multiple starred names in sequence pattern', pattern)
                star_wildcard = self._wildcard_star_check(pattern)
                only_wildcard = only_wildcard and star_wildcard
                star = i
                continue
            only_wildcard = only_wildcard and self._wildcard_check(pattern)
        pc.on_top += 1
        self.emit('MATCH_SEQUENCE')
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        if star < 0:
            self.emit('GET_LEN')
            self.emit('LOAD_CONST', size)
            self.emit('COMPARE_OP', '==')
            self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        elif size > 1:
            self.emit('GET_LEN')
            self.emit('LOAD_CONST', size - 1)
            self.emit('COMPARE_OP', '>=')
            self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        pc.on_top -= 1
        if only_wildcard:
            self.emit('POP_TOP')
        elif star_wildcard:
            self._pattern_helper_sequence_subscr(patterns, star, pc)
        else:
            self._pattern_helper_sequence_unpack(patterns, star, pc)

    def _pattern_helper_sequence_unpack(self, patterns: list[ast.pattern], star: int, pc: PatternContext) -> None:
        if False:
            i = 10
            return i + 15
        self._pattern_unpack_helper(patterns)
        size = len(patterns)
        pc.on_top += size
        for pattern in patterns:
            pc.on_top -= 1
            self._visit_subpattern(pattern, pc)

    def _pattern_helper_sequence_subscr(self, patterns: list[ast.pattern], star: int, pc: PatternContext) -> None:
        if False:
            return 10
        '\n        Like _pattern_helper_sequence_unpack, but uses BINARY_SUBSCR instead of\n        UNPACK_SEQUENCE / UNPACK_EX. This is more efficient for patterns with a\n        starred wildcard like [first, *_] / [first, *_, last] / [*_, last] / etc.\n        '
        pc.on_top += 1
        size = len(patterns)
        for (i, pattern) in enumerate(patterns):
            if self._wildcard_check(pattern):
                continue
            if i == star:
                assert self._wildcard_star_check(pattern)
                continue
            self.emit('DUP_TOP')
            if i < star:
                self.emit('LOAD_CONST', i)
            else:
                self.emit('GET_LEN')
                self.emit('LOAD_CONST', size - i)
                self.emit('BINARY_SUBTRACT')
            self.emit('BINARY_SUBSCR')
            self._visit_subpattern(pattern, pc)
        pc.on_top -= 1
        self.emit('POP_TOP')

    def _pattern_unpack_helper(self, elts: list[ast.pattern]) -> None:
        if False:
            i = 10
            return i + 15
        n = len(elts)
        seen_star = 0
        for (i, elt) in enumerate(elts):
            if isinstance(elt, ast.MatchStar) and (not seen_star):
                if i >= 1 << 8 or n - i - 1 >= INT_MAX >> 8:
                    raise self.syntax_error('too many expressions in star-unpacking sequence pattern', elt)
                self.emit('UNPACK_EX', i + (n - i - 1 << 8))
                seen_star = 1
            elif isinstance(elt, ast.MatchStar):
                raise self.syntax_error('multiple starred expressions in sequence pattern', elt)
        if not seen_star:
            self.emit('UNPACK_SEQUENCE', n)

    def _wildcard_check(self, pattern: ast.pattern) -> bool:
        if False:
            print('Hello World!')
        return isinstance(pattern, ast.MatchAs) and (not pattern.name)

    def _wildcard_star_check(self, pattern: ast.pattern) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return isinstance(pattern, ast.MatchStar) and (not pattern.name)

    def _visit_subpattern(self, pattern: ast.pattern, pc: PatternContext) -> None:
        if False:
            print('Hello World!')
        'Visit a pattern, but turn off checks for irrefutability.\n\n        See compiler_pattern_subpattern in compile.c\n        '
        allow_irrefutable = pc.allow_irrefutable
        pc.allow_irrefutable = True
        try:
            self.visit(pattern, pc)
        finally:
            pc.allow_irrefutable = allow_irrefutable

    def visitMatchMapping(self, node: ast.MatchMapping, pc: PatternContext) -> None:
        if False:
            while True:
                i = 10
        keys = node.keys
        patterns = node.patterns
        size = len(keys)
        npatterns = len(patterns)
        if size != npatterns:
            raise self.syntax_error(f'keys ({size}) / patterns ({npatterns}) length mismatch in mapping pattern', node)
        star_target = node.rest
        pc.on_top += 1
        self.emit('MATCH_MAPPING')
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        if not size and (not star_target):
            pc.on_top -= 1
            self.emit('POP_TOP')
            return
        if size:
            self.emit('GET_LEN')
            self.emit('LOAD_CONST', size)
            self.emit('COMPARE_OP', '>=')
            self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        if INT_MAX < size - 1:
            raise self.syntax_error('too many sub-patterns in mapping pattern', node)
        seen = set()
        for (i, key) in enumerate(keys):
            if key is None:
                raise self.syntax_error("Can't use NULL keys in MatchMapping (set 'rest' parameter instead)", patterns[i])
            if isinstance(key, ast.Constant):
                if key.value in seen:
                    raise self.syntax_error(f'mapping pattern checks duplicate key ({key.value})', node)
                seen.add(key.value)
            elif not isinstance(key, ast.Attribute):
                raise self.syntax_error('mapping pattern keys may only match literals and attribute lookups', node)
            self.visit(key)
        self.emit('BUILD_TUPLE', size)
        self.emit('MATCH_KEYS')
        pc.on_top += 2
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        for (i, pattern) in enumerate(patterns):
            if self._wildcard_check(pattern):
                continue
            self.emit('DUP_TOP')
            self.emit('LOAD_CONST', i)
            self.emit('BINARY_SUBSCR')
            self._visit_subpattern(pattern, pc)
        pc.on_top -= 2
        self.emit('POP_TOP')
        if star_target:
            self.emit('COPY_DICT_WITHOUT_KEYS')
            self._pattern_helper_store_name(star_target, pc, node)
        else:
            self.emit('POP_TOP')
        pc.on_top -= 1
        self.emit('POP_TOP')

    def visitMatchClass(self, node: ast.MatchClass, pc: PatternContext) -> None:
        if False:
            while True:
                i = 10
        patterns = node.patterns
        kwd_attrs = node.kwd_attrs
        kwd_patterns = node.kwd_patterns
        nargs = len(patterns)
        nattrs = len(kwd_attrs)
        nkwd_patterns = len(kwd_patterns)
        if nattrs != nkwd_patterns:
            raise self.syntax_error(f'kwd_attrs ({nattrs}) / kwd_patterns ({nkwd_patterns}) length mismatch in class pattern', node)
        if INT_MAX < nargs or INT_MAX < nargs + nattrs - 1:
            raise self.syntax_error(f'too many sub-patterns in class pattern {node.cls}', node)
        if nattrs:
            self._validate_kwd_attrs(kwd_attrs, kwd_patterns)
        self.visit(node.cls)
        attr_names = tuple((name for name in kwd_attrs))
        self.emit('LOAD_CONST', attr_names)
        self.emit('MATCH_CLASS', nargs)
        pc.on_top += 1
        self._jump_to_fail_pop(pc, 'POP_JUMP_IF_FALSE')
        for i in range(nargs + nattrs):
            if i < nargs:
                pattern = patterns[i]
            else:
                pattern = kwd_patterns[i - nargs]
            if self._wildcard_check(pattern):
                continue
            self.emit('DUP_TOP')
            self.emit('LOAD_CONST', i)
            self.emit('BINARY_SUBSCR')
            self._visit_subpattern(pattern, pc)
        pc.on_top -= 1
        self.emit('POP_TOP')

    def _validate_kwd_attrs(self, attrs: list[str], patterns: list[ast.pattern]) -> None:
        if False:
            print('Hello World!')
        nattrs = len(attrs)
        for (i, attr) in enumerate(attrs):
            pattern = patterns[i]
            self.set_lineno(pattern)
            self._check_forbidden_name(attr, ast.Store, pattern)
            for j in range(i + 1, nattrs):
                other = attrs[j]
                if attr == other:
                    self.set_lineno(patterns[j])
                    raise self.syntax_error(f'attribute name repeated in class pattern: {attr}', patterns[j])

    def visitMatchStar(self, node: ast.MatchStar, pc: PatternContext) -> None:
        if False:
            while True:
                i = 10
        self._pattern_helper_store_name(node.name, pc, node)

    def visitMatchAs(self, node: ast.MatchAs, pc: PatternContext) -> None:
        if False:
            return 10
        'See compiler_pattern_as in compile.c'
        pat = node.pattern
        if pat is None:
            if not pc.allow_irrefutable:
                if node.name:
                    raise SyntaxError(f'name capture {node.name!r} makes remaining patterns unreachable')
                raise SyntaxError('wildcard makes remaining patterns unreachable')
            self._pattern_helper_store_name(node.name, pc, node)
            return
        pc.on_top += 1
        self.emit('DUP_TOP')
        self.visit(pat, pc)
        pc.on_top -= 1
        self._pattern_helper_store_name(node.name, pc, node)

    def visitMatchOr(self, node: ast.MatchOr, pc: PatternContext) -> None:
        if False:
            return 10
        'See compiler_pattern_or in compile.c'
        end = self.newBlock('match_or_end')
        size = len(node.patterns)
        assert size > 1
        old_pc = pc
        pc = pc.clone()
        control: list[str] | None = None
        for alt in node.patterns:
            self.set_lineno(alt)
            pc.stores = []
            pc.allow_irrefutable = alt is node.patterns[-1] and old_pc.allow_irrefutable
            pc.fail_pop = []
            pc.on_top = 0
            self.emit('DUP_TOP')
            self.visit(alt, pc)
            nstores = len(pc.stores)
            if alt is node.patterns[0]:
                assert control is None
                control = pc.stores
            elif nstores != len(control or []):
                raise SyntaxError('alternative patterns bind different names')
            elif nstores:
                icontrol = nstores
                assert control is not None
                while icontrol:
                    icontrol -= 1
                    name = control[icontrol]
                    istores = pc.stores.index(name)
                    if icontrol != istores:
                        assert istores < icontrol
                        rotations = istores + 1
                        rotated = pc.stores[:rotations]
                        del pc.stores[:rotations]
                        pc.stores[icontrol - istores:icontrol - istores] = rotated
                        for _ in range(rotations):
                            self.emit('ROT_N', icontrol + 1)
            assert control is not None
            self.emit('JUMP_FORWARD', end)
            self.nextBlock()
            self._emit_and_reset_fail_pop(pc)
        assert control is not None
        pc = old_pc
        self.emit('POP_TOP')
        self._jump_to_fail_pop(pc, 'JUMP_FORWARD')
        self.nextBlock(end)
        nstores = len(control)
        nrots = nstores + 1 + pc.on_top + len(pc.stores)
        for i in range(nstores):
            self.emit('ROT_N', nrots)
            name = control[i]
            dupe = name in pc.stores
            if dupe:
                raise self._error_duplicate_store(name)
            pc.stores.append(name)
        self.emit('POP_TOP')

    def _pattern_helper_store_name(self, name: str | None, pc: PatternContext, loc: ast.AST) -> None:
        if False:
            i = 10
            return i + 15
        if name is None:
            self.emit('POP_TOP')
            return
        self._check_forbidden_name(name, ast.Store, loc)
        duplicate = name in pc.stores
        if duplicate:
            raise self._error_duplicate_store(name)
        self.emit('ROT_N', pc.on_top + len(pc.stores) + 1)
        pc.stores.append(name)

    def _check_forbidden_name(self, name: str, ctx: type[ast.expr_context], loc: ast.AST) -> None:
        if False:
            i = 10
            return i + 15
        if ctx is ast.Store and name == '__debug__':
            raise SyntaxError('cannot assign to __debug__', loc)
        if ctx is ast.Del and name == '__debug__':
            raise SyntaxError('cannot delete __debug__', loc)

    def _error_duplicate_store(self, name: str) -> SyntaxError:
        if False:
            while True:
                i = 10
        return SyntaxError(f'multiple assignments to name {name!r} in pattern')

    def _ensure_fail_pop(self, pc: PatternContext, n: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Resize pc.fail_pop to allow for n items to be popped on failure.'
        size = n + 1
        if size <= len(pc.fail_pop):
            return
        while len(pc.fail_pop) < size:
            pc.fail_pop.append(self.newBlock(f'match_fail_pop_{len(pc.fail_pop)}'))

    def _jump_to_fail_pop(self, pc: PatternContext, op: str) -> None:
        if False:
            return 10
        'Use op to jump to the correct fail_pop block.'
        pops = pc.on_top + len(pc.stores)
        self._ensure_fail_pop(pc, pops)
        self.emit(op, pc.fail_pop[pops])
        self.nextBlock()

    def _emit_and_reset_fail_pop(self, pc: PatternContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Build all of the fail_pop blocks and reset fail_pop.'
        if not pc.fail_pop:
            self.nextBlock()
        while pc.fail_pop:
            self.nextBlock(pc.fail_pop.pop())
            if pc.fail_pop:
                self.emit('POP_TOP')

    def unwind_setup_entry(self, e: Entry, preserve_tos: int) -> None:
        if False:
            while True:
                i = 10
        if e.kind in (WHILE_LOOP, EXCEPTION_HANDLER, ASYNC_COMPREHENSION_GENERATOR):
            return
        elif e.kind == FOR_LOOP:
            if preserve_tos:
                self.emit('ROT_TWO')
            self.emit('POP_TOP')
        elif e.kind == TRY_EXCEPT:
            self.emit('POP_BLOCK')
        elif e.kind == FINALLY_TRY:
            self.emit('POP_BLOCK')
            if preserve_tos:
                self.setups.push(Entry(POP_VALUE, None, None, None))
            assert callable(e.unwinding_datum)
            e.unwinding_datum()
            if preserve_tos:
                self.setups.pop()
            self.set_no_lineno()
        elif e.kind == FINALLY_END:
            if preserve_tos:
                self.emit('ROT_FOUR')
            self.emit('POP_TOP')
            self.emit('POP_TOP')
            self.emit('POP_TOP')
            if preserve_tos:
                self.emit('ROT_FOUR')
            self.emit('POP_EXCEPT')
        elif e.kind in (WITH, ASYNC_WITH):
            assert isinstance(e.unwinding_datum, AST)
            self.set_lineno(e.unwinding_datum)
            self.emit('POP_BLOCK')
            if preserve_tos:
                self.emit('ROT_TWO')
            self.call_exit_with_nones()
            if e.kind == ASYNC_WITH:
                self.emit('GET_AWAITABLE')
                self.emit('LOAD_CONST', None)
                self.emit('YIELD_FROM')
            self.emit('POP_TOP')
            self.set_no_lineno()
        elif e.kind == HANDLER_CLEANUP:
            datum = e.unwinding_datum
            if datum is not None:
                self.emit('POP_BLOCK')
            if preserve_tos:
                self.emit('ROT_FOUR')
            self.emit('POP_EXCEPT')
            if datum is not None:
                self.emit('LOAD_CONST', None)
                self.storeName(datum)
                self.delName(datum)
        elif e.kind == POP_VALUE:
            if preserve_tos:
                self.emit('ROT_TWO')
            self.emit('POP_TOP')
        else:
            raise Exception(f'Unexpected kind {e.kind}')

    def unwind_setup_entries(self, preserve_tos: bool, stop_on_loop: bool=False) -> Entry | None:
        if False:
            while True:
                i = 10
        if len(self.setups) == 0:
            return None
        top = self.setups[-1]
        if stop_on_loop and top.kind in (WHILE_LOOP, FOR_LOOP):
            return top
        copy = self.setups.pop()
        self.unwind_setup_entry(copy, preserve_tos)
        loop = self.unwind_setup_entries(preserve_tos, stop_on_loop)
        self.setups.push(copy)
        return loop

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.tree, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            return self.tree.name
        elif isinstance(self.tree, ast.SetComp):
            return '<setcomp>'
        elif isinstance(self.tree, ast.ListComp):
            return '<listcomp>'
        elif isinstance(self.tree, ast.DictComp):
            return '<dictcomp>'
        elif isinstance(self.tree, ast.GeneratorExp):
            return '<genexpr>'
        elif isinstance(self.tree, ast.Lambda):
            return '<lambda>'

    def finishFunction(self):
        if False:
            return 10
        if self.graph.current.returns:
            return
        if not isinstance(self.tree, ast.Lambda):
            self.set_no_lineno()
            self.emit('LOAD_CONST', None)
        self.emit('RETURN_VALUE')

    def make_child_codegen(self, tree: FuncOrLambda | CompNode | ast.ClassDef, graph: PyFlowGraph, codegen_type: Optional[Type[CodeGenerator]]=None) -> CodeGenerator:
        if False:
            while True:
                i = 10
        if codegen_type is None:
            codegen_type = type(self)
        return codegen_type(self, tree, self.symbols, graph, flags=self.flags, optimization_lvl=self.optimization_lvl)

    def make_func_codegen(self, func: FuncOrLambda | CompNode, func_args: ast.arguments, name: str, first_lineno: int) -> CodeGenerator:
        if False:
            while True:
                i = 10
        filename = self.graph.filename
        symbols = self.symbols
        class_name = self.class_name
        graph = self.make_function_graph(func, func_args, filename, symbols.scopes, class_name, name, first_lineno)
        graph.emit_gen_start()
        res = self.make_child_codegen(func, graph)
        res.optimized = 1
        res.class_name = class_name
        return res

    def make_class_codegen(self, klass: ast.ClassDef, first_lineno: int) -> CodeGenerator:
        if False:
            i = 10
            return i + 15
        filename = self.graph.filename
        symbols = self.symbols
        scope = symbols.scopes[klass]
        graph = self.flow_graph(klass.name, filename, scope, optimized=0, klass=True, docstring=self.get_docstring(klass), firstline=first_lineno)
        res = self.make_child_codegen(klass, graph)
        res.class_name = klass.name
        return res

    def make_function_graph(self, func: FuncOrLambda | CompNode, func_args: ast.arguments, filename: str, scopes, class_name: str, name: str, first_lineno: int) -> PyFlowGraph:
        if False:
            print('Hello World!')
        args = [misc.mangle(elt.arg, class_name) for elt in itertools.chain(func_args.posonlyargs, func_args.args)]
        kwonlyargs = [misc.mangle(elt.arg, class_name) for elt in func_args.kwonlyargs]
        starargs = []
        if (va := func_args.vararg):
            starargs.append(va.arg)
        if (kw := func_args.kwarg):
            starargs.append(kw.arg)
        scope = scopes[func]
        graph = self.flow_graph(name, filename, scope, flags=self.get_graph_flags(func, func_args, scope), args=args, kwonlyargs=kwonlyargs, starargs=starargs, optimized=1, docstring=self.get_docstring(func), firstline=first_lineno, posonlyargs=len(func_args.posonlyargs))
        return graph

    def get_docstring(self, func: ast.Module | ast.ClassDef | FuncOrLambda | CompNode) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        doc = None
        if not isinstance(func, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and (not self.strip_docstrings):
            doc = get_docstring(func)
        return doc

    def get_graph_flags(self, func: FuncOrLambda | CompNode, func_args: ast.arguments, scope: Scope) -> int:
        if False:
            for i in range(10):
                print('nop')
        flags = 0
        if func_args.vararg:
            flags = flags | CO_VARARGS
        if func_args.kwarg:
            flags = flags | CO_VARKEYWORDS
        if scope.nested:
            flags = flags | CO_NESTED
        if scope.generator and (not scope.coroutine):
            flags = flags | CO_GENERATOR
        if not scope.generator and scope.coroutine:
            flags = flags | CO_COROUTINE
        if scope.generator and scope.coroutine:
            flags = flags | CO_ASYNC_GENERATOR
        return flags

    @classmethod
    def make_code_gen(cls, module_name: str, tree: ast.Module, filename: str, flags: int, optimize: int, ast_optimizer_enabled: bool=True):
        if False:
            print('Hello World!')
        future_flags = find_futures(flags, tree)
        if ast_optimizer_enabled:
            tree = cls.optimize_tree(optimize, tree, bool(future_flags & consts.CO_FUTURE_ANNOTATIONS))
        s = cls._SymbolVisitor(future_flags)
        walk(tree, s)
        graph = cls.flow_graph(module_name, filename, s.scopes[tree], firstline=1)
        code_gen = cls(None, tree, s, graph, flags, optimize, future_flags=future_flags)
        code_gen._qual_name = module_name
        walk(tree, code_gen)
        return code_gen

    @classmethod
    def optimize_tree(cls, optimize: int, tree: AST, string_anns: bool):
        if False:
            print('Hello World!')
        return AstOptimizer(optimize=optimize > 0, string_anns=string_anns).visit(tree)

    def visit(self, node: Union[Sequence[AST], AST], *args):
        if False:
            return 10
        old_lineno = None
        if isinstance(node, ast.expr):
            old_lineno = self.graph.lineno
            self.set_lineno(node)
        elif isinstance(node, (ast.stmt, ast.pattern)):
            self.set_lineno(node)
        ret = super().visit(node, *args)
        if old_lineno is not None and old_lineno != self.graph.lineno:
            self.graph.lineno = old_lineno
        return ret

    def visitStatements(self, nodes: Sequence[AST], *args) -> None:
        if False:
            for i in range(10):
                print('nop')
        for node in nodes:
            self.visit(node, *args)

class Entry:
    kind: int
    block: pyassem.Block
    exit: Optional[pyassem.Block]
    unwinding_datum: object

    def __init__(self, kind, block, exit, unwinding_datum):
        if False:
            i = 10
            return i + 15
        self.kind = kind
        self.block = block
        self.exit = exit
        self.unwinding_datum = unwinding_datum

class CinderCodeGenerator(CodeGenerator):
    """
    Code generator equivalent to `Python/compile.c` in Cinder.

    The base `CodeGenerator` is equivalent to upstream `Python/compile.c`.
    """
    flow_graph = pyassem.PyFlowGraphCinder
    _SymbolVisitor = symbols.CinderSymbolVisitor

    def compile_comprehension(self, node: CompNode, name: str, elt: ast.expr, val: ast.expr | None, opcode: str, oparg: object=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        scope = self.scopes[node]
        if scope.inlined:
            gen = self
        else:
            gen = self.make_func_codegen(node, self.conjure_arguments([ast.arg('.0', None)]), name, node.lineno)
        gen.set_lineno(node)
        if opcode:
            gen.emit(opcode, oparg)
        gen.compile_comprehension_generator(node.generators, 0, 0, elt, val, type(node), not scope.inlined)
        if scope.inlined:
            to_delete = [v for v in scope.defs if v != '.0' and v not in scope.nonlocals and (v not in scope.parent.cells)]
            to_delete.sort()
            for v in to_delete:
                self.delName(v)
            return
        if not isinstance(node, ast.GeneratorExp):
            gen.emit('RETURN_VALUE')
        gen.finishFunction()
        self._makeClosure(gen, 0)
        self.visit(node.generators[0].iter)
        if node.generators[0].is_async:
            self.emit('GET_AITER')
        else:
            self.emit('GET_ITER')
        self.emit('CALL_FUNCTION', 1)
        if gen.scope.coroutine and type(node) is not ast.GeneratorExp:
            self.emit('GET_AWAITABLE')
            self.emit('LOAD_CONST', None)
            self.emit('YIELD_FROM')

    def set_qual_name(self, qualname):
        if False:
            for i in range(10):
                print('nop')
        self._qual_name = qualname

    def getCode(self):
        if False:
            i = 10
            return i + 15
        code = super().getCode()
        _set_qualname(code, self._qual_name)
        return code

    def _is_super_call(self, node):
        if False:
            print('Hello World!')
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'super' or node.keywords:
            return False
        if self.scope.check_name('super') != SC_GLOBAL_IMPLICIT or (self.module_gen.scope.check_name('super') != SC_GLOBAL_IMPLICIT and self.module_gen.scope.check_name('super') != SC_LOCAL):
            return False
        if len(node.args) == 2:
            return True
        if len(node.args) == 0:
            if len(self.scope.params) == 0:
                return False
            return self.scope.check_name('__class__') == SC_FREE
        return False

    def _emit_args_for_super(self, super_call, attr):
        if False:
            i = 10
            return i + 15
        if len(super_call.args) == 0:
            self.loadName('__class__')
            self.loadName(next(iter(self.scope.params)))
        else:
            for arg in super_call.args:
                self.visit(arg)
        return (self.mangle(attr), len(super_call.args) == 0)

    def visitAttribute(self, node):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node.ctx, ast.Load) and self._is_super_call(node.value):
            self.emit('LOAD_GLOBAL', 'super')
            load_arg = self._emit_args_for_super(node.value, node.attr)
            self.emit('LOAD_ATTR_SUPER', load_arg)
        else:
            super().visitAttribute(node)

    def visitCall(self, node):
        if False:
            print('Hello World!')
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.ctx, ast.Load) or node.keywords or any((isinstance(arg, ast.Starred) for arg in node.args)) or (len(node.args) >= STACK_USE_GUIDELINE) or (not self._is_super_call(node.func.value)):
            return super().visitCall(node)
        with self.temp_lineno(node.func.end_lineno):
            self.emit('LOAD_GLOBAL', 'super')
            load_arg = self._emit_args_for_super(node.func.value, node.func.attr)
            self.emit('LOAD_METHOD_SUPER', load_arg)
            for arg in node.args:
                self.visit(arg)
            self.emit('CALL_METHOD', len(node.args))

def get_default_generator():
    if False:
        while True:
            i = 10
    if 'cinder' in sys.version:
        return CinderCodeGenerator
    return CodeGenerator

def get_docstring(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    if False:
        print('Hello World!')
    if node.body and (b0 := node.body[0]) and isinstance(b0, ast.Expr) and (b0v := b0.value) and isinstance(b0v, ast.Str):
        return b0v.s

def findOp(node):
    if False:
        return 10
    'Find the op (DELETE, LOAD, STORE) in an AssTuple tree'
    v = OpFinder()
    v.VERBOSE = 0
    walk(node, v)
    return v.op

class OpFinder:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = None

    def visitAssName(self, node):
        if False:
            print('Hello World!')
        if self.op is None:
            self.op = node.flags
        elif self.op != node.flags:
            raise ValueError('mixed ops in stmt')
    visitAssAttr = visitAssName
    visitSubscript = visitAssName
PythonCodeGenerator = get_default_generator()
if __name__ == '__main__':
    for file in sys.argv[1:]:
        compileFile(file)