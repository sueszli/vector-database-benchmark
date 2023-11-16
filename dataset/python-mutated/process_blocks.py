"""Analyze code blocks and process opcodes."""
from pytype.blocks import blocks
from pytype.pyc import opcodes
from pytype.pyc import pyc
CODE_LOADING_OPCODES = (opcodes.LOAD_CONST,)

def _is_function_def(fn_code):
    if False:
        i = 10
        return i + 15
    'Helper function for CollectFunctionTypeCommentTargetsVisitor.'
    first = fn_code.name[0]
    if not (first == '_' or first.isalpha()):
        return False
    op = fn_code.get_first_opcode()
    if isinstance(op, opcodes.LOAD_NAME) and op.argval == '__name__':
        return False
    return True

class CollectAnnotationTargetsVisitor:
    """Collect opcodes that might have annotations attached."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.store_ops = {}
        self.make_function_ops = {}

    def visit_code(self, code):
        if False:
            print('Hello World!')
        'Find STORE_* and MAKE_FUNCTION opcodes for attaching annotations.'
        offset = 1 if code.python_version >= (3, 11) else 2
        co_code = list(code.code_iter)
        for (i, op) in enumerate(co_code):
            if isinstance(op, opcodes.MAKE_FUNCTION):
                code_op = co_code[i - offset]
                assert isinstance(code_op, CODE_LOADING_OPCODES), code_op.__class__
                fn_code = code.consts[code_op.arg]
                if not _is_function_def(fn_code):
                    continue
                end_line = min((op.line for op in fn_code.code_iter if not isinstance(op, opcodes.RESUME)))
                self.make_function_ops[op.line] = (end_line, op)
            elif isinstance(op, blocks.STORE_OPCODES) and op.line not in self.make_function_ops:
                self.store_ops[op.line] = op
        return code

class FunctionDefVisitor:
    """Add metadata to function definition opcodes."""

    def __init__(self, param_annotations):
        if False:
            while True:
                i = 10
        self.annots = param_annotations

    def visit_code(self, code):
        if False:
            print('Hello World!')
        for op in code.code_iter:
            if isinstance(op, opcodes.MAKE_FUNCTION):
                if op.line in self.annots:
                    op.metadata.signature_annotations = self.annots[op.line]
        return code

def merge_annotations(code, annotations, param_annotations):
    if False:
        i = 10
        return i + 15
    'Merges type comments into their associated opcodes.\n\n  Modifies code in place.\n\n  Args:\n    code: An OrderedCode object.\n    annotations: A map of lines to annotations.\n    param_annotations: A list of _ParamAnnotations from the director\n\n  Returns:\n    The code with annotations added to the relevant opcodes.\n  '
    if param_annotations:
        visitor = FunctionDefVisitor(param_annotations)
        pyc.visit(code, visitor)
    visitor = CollectAnnotationTargetsVisitor()
    code = pyc.visit(code, visitor)
    for (line, op) in visitor.store_ops.items():
        if line in annotations:
            annot = annotations[line]
            if annot.name in (None, op.argval):
                op.annotation = annot.annotation
    for (start, (end, op)) in sorted(visitor.make_function_ops.items(), reverse=True):
        for i in range(start, end):
            if i in annotations:
                op.annotation = (annotations[i].annotation, i)
                break
    return code

def adjust_returns(code, block_returns):
    if False:
        i = 10
        return i + 15
    'Adjust line numbers for return statements in with blocks.'
    rets = {k: iter(v) for (k, v) in block_returns}
    for block in code.order:
        for op in block:
            if op.__class__.__name__ == 'RETURN_VALUE':
                if op.line in rets:
                    lines = rets[op.line]
                    new_line = next(lines, None)
                    if new_line:
                        op.line = new_line

def check_out_of_order(code):
    if False:
        for i in range(10):
            print('nop')
    'Check if a line of code is executed out of order.'
    last_line = []
    for block in code.order:
        for op in block:
            if not last_line or last_line[-1].line == op.line:
                last_line.append(op)
            else:
                if op.line < last_line[-1].line:
                    for x in last_line:
                        x.metadata.is_out_of_order = True
                last_line = [op]